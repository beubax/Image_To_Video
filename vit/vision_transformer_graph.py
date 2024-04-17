"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.io import write_jpeg
from torchvision.transforms import transforms as T
from einops import rearrange
from torchvision.utils import flow_to_image
from torchvision.transforms._transforms_video import ToTensorVideo
import numpy as np
import spconv.pytorch as spconv
from torchvision.models.optical_flow import raft_large
import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as F
from torch.masked import masked_tensor
from vit.graph_transformer_pytorch import GraphTransformer
from pytorchvideo.transforms import Normalize
from vit.utils import create_mask, trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., temporal_attn_mask=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temporal_attn_mask = temporal_attn_mask
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, pad_mask = None, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.temporal_attn_mask is not None:
            attn[self.temporal_attn_mask.expand_as(attn)] = -1e9

        if pad_mask is not None:
            attn[pad_mask.unsqueeze(1).expand_as(attn)] = -1e9

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, qkv

class Spatial_Weighting(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        key = qkv[1].clone().detach()
        key = rearrange(key, '(b t) h n d -> b (t n) (h d)', t=self.num_frames)
        B, _, _ = key.shape
        (keys1, keys2) = torch.split(key, B//2, dim=0)
        feats1 = keys1 @ keys1.transpose(-1, -2)
        feats2 = keys2 @ keys2.transpose(-1, -2)
        feats = feats1 * 0.5 + feats2 * 0.5
        feats = feats > 0.2
        feats = torch.where(feats.type(torch.cuda.FloatTensor) == 0, 1e-5, feats)
        d_i = torch.sum(feats, dim=-1)
        D = torch.diag_embed(d_i)
        _, eigenvectors = torch.lobpcg(A=D-feats, B=D, k=2, largest=False)
        eigenvec = eigenvectors[:, :, 1]
        avg = torch.mean(eigenvec, dim=-1).unsqueeze(-1)
        bipartition = torch.gt(eigenvec , 0)
        bipartition = torch.where(avg > 0,bipartition, torch.logical_not(bipartition)) 
        bipartition = bipartition.to(torch.float)
        eigenvec = torch.abs(torch.mul(eigenvec, bipartition))        
        # _, indices = torch.topk(eigenvec, 1000, dim=1)
        # mask = torch.zeros_like(eigenvec)
        # mask.scatter_(1, indices, 1)
        # eigenvec = eigenvec * mask
        eigenvec[eigenvec == 0] = float("-inf")
        eigenvec = self.softmax(eigenvec)
        return eigenvec


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., num_frames = 16, act_layer=nn.GELU, norm_layer=nn.LayerNorm, temporal_attn_mask=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, temporal_attn_mask=temporal_attn_mask)
        self.spatial_weighting = Spatial_Weighting(num_frames)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pad_mask = None, return_spatial_map=False, return_attention=False, register_hook=False):
        y, attn, qkv = self.attn(self.norm1(x), pad_mask=pad_mask, register_hook=register_hook)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_spatial_map:
            spatial_map = self.spatial_weighting(qkv)
            return x, spatial_map
        return x
    
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # linear layer
        return self.linear(x)
    
class ExampleNet(nn.Module):
    def __init__(self, temporal_embed_dim):
        super().__init__()
        self.temporal_embed_dim = temporal_embed_dim
        self.model = nn.Sequential(
        spnn.Conv3d(768, self.temporal_embed_dim, kernel_size=(2,3,3), padding=(0,0,0), stride=(2,3,3)),
    )

    def forward(self, x):
        return self.model(x)# .dense()
    
# class ExampleNet(nn.Module):
#     def __init__(self, temporal_embed_dim):
#         super().__init__()
#         self.temporal_embed_dim = temporal_embed_dim
#         self.net = nn.Sequential(
#             nn.Conv3d(768, self.temporal_embed_dim, kernel_size=(2,3,3), padding=(0,0,0), stride=(2,3,3)),
#         )

#     def forward(self, x):
#         return self.net(x)# .dense()


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., temporal_depth=2, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., num_frames = 16, norm_layer=nn.LayerNorm, temporal_embed_dim = 512, temporal_num_heads = 8, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 8, temporal_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.temporal_attn_mask = create_mask(size=128, block_size=16)
        # self.temporal_norm = nn.LayerNorm()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, temporal_attn_mask=None)
            for i in range(depth)])
        self.temporal_blocks = nn.ModuleList([
            Block(
                dim=temporal_embed_dim, num_heads=temporal_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, num_frames = self.num_frames, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, temporal_attn_mask=self.temporal_attn_mask)
            for i in range(temporal_depth)])
        # self.norm = norm_layer(embed_dim)
        self.temporal_norm = norm_layer(temporal_embed_dim)
        self.point_cloud_tokenize = ExampleNet(temporal_embed_dim)
        # Classifier head
        self.head = LinearClassifier(temporal_embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.graph_transformer = GraphTransformer(dim = temporal_embed_dim,
        # depth = 6,
        # with_feedforwards = True,
        # gated_residual = True,
        # accept_adjacency_matrix = True 
        # )
        self.transforms = train_transform = T.Compose([ToTensorVideo(),  # C, T, H, W
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.flow_model = raft_large(pretrained=True, progress=False)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def prepare_tokens(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        # cls_tokens = self.cls_token.expand(B * T, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, W, H)

        n = x.shape[1]
        # x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        # x = x + self.temporal_embedding
        # x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        return self.pos_drop(x)

    def forward(self, x, register_hook = False):
        B, C, T, H, W = x.shape
        with torch.no_grad():
            flow_video = x.permute(0, 2, 1, 3, 4)
            flow_video2 = flow_video[:, 1:]
            flow_video2 = torch.cat((flow_video2, flow_video2[:, -2].unsqueeze(1)), dim=1)
            flow_video = rearrange(flow_video, 'b t c h w -> (b t) c h w')
            flow_video2 = rearrange(flow_video2, 'b t c h w -> (b t) c h w')
            list_of_flows = self.flow_model(flow_video, flow_video2)
            predicted_flow = list_of_flows[-1]
            flow_img = flow_to_image(predicted_flow)
            for i, img in enumerate(flow_img):
                output_folder = "output/"  # Update this to the folder of your choice
                write_jpeg(img.to("cpu"), output_folder + f"predicted_flow_{i}.jpg")
            flow_video = self.transforms(flow_img.permute(0, 2, 3, 1))
            flow_video = rearrange(flow_video, 'c (b t) h w -> b c t h w', t=self.num_frames)
            x = torch.cat((x, flow_video), dim=0)
            x = self.prepare_tokens(x)
            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x, register_hook=register_hook)
                else:
                    # return spatial_map of the last block
                    x, spatial_map =  blk(x, return_spatial_map=True, register_hook=register_hook)

        x = rearrange(x, '(b t) n d -> b (t n) d', t=self.num_frames)
        (x, _) = torch.split(x, B, dim=0)
        # nodes = torch.randn(2, 128, 256)
        # adj_mat = torch.randint(0, 2, (2, 128, 128))
        # mask = torch.ones(2, 128).bool()

        # nodes, edges = self.graph_transformer(nodes=x, adj_mat = feats)
        # print(nodes.shape)
        spatial_map = spatial_map.unsqueeze(2)
        x = torch.einsum('ijl,ijp->ijp', spatial_map, x)
        # # x = rearrange(x, 'n t (h w) c -> n c t h w', h=14)
        x = rearrange(x, 'n (t h w) c -> n t h w c', t = self.num_frames ,h=14) 
        spatial_map = spatial_map.squeeze(2)
        spatial_map = rearrange(spatial_map, 'n (t h w) -> n t h w', t=self.num_frames, h=14)
        indices = torch.nonzero(spatial_map)
        feats = x[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
        sparse_tensor = SparseTensor(coords=indices.to(torch.int32), feats=feats)
        x = self.point_cloud_tokenize(sparse_tensor)
        tensor = torch.zeros(size=(B, 8, 4, 4, 512)).cuda()
        tensor[x.coords[:, 0], x.coords[:, 1], x.coords[:, 2], x.coords[:, 3]] = x.feats
        x = rearrange(tensor, 'b t h w c -> (b h w) t c')
        x = x + self.temporal_embedding
        x = rearrange(x, '(b h w) t c -> b (t h w) c', b = B, h = 4, w = 4)
        mask = x.mean(dim=-1) == 0
        mask = mask.unsqueeze(1).expand((x.shape[0], x.shape[1], x.shape[1]))
        for i, blk in enumerate(self.temporal_blocks):
            x = blk(x, pad_mask = mask, register_hook=register_hook)
        
        x = self.temporal_norm(x)
        x = torch.mean(x, dim=1)
        x = self.head(x)
        return x


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(num_classes, patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(num_classes, patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# for name, param in model.named_parameters():
#    print('{}: {}'.format(name, param.requires_grad))
# num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
# num_total_param = sum(p.numel() for p in model.parameters())
# print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))