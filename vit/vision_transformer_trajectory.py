"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import spconv.pytorch as spconv
import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as F
from torch.masked import masked_tensor
import orthoformer_helper

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

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x, seq_len=196, num_frames=16, approx="orthoformer", num_landmarks=128, pad_mask = None, register_hook=False):
        b, n, c = x.shape
        P = seq_len
        F = num_frames
        h = self.num_heads
        B = x.shape[0] // num_frames
        N = seq_len * num_frames
        C = x.shape[-1]

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
         # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, '(b t) h n d -> (b h) (t n) d', h=h, t=F), (q, k, v))
        
        if approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q, k, v,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        else:
            # Using full attention
            q_dot_k = q @ k.transpose(-2, -1)
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) #For each token, compute contribution of all other tokens per frame
            attn = self.attn_drop(space_attn)                   #We are applying softmax per frame for each token
            v_ = rearrange(v, 'b (f n) d -> b f n d', f=F, n=P)
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) #This calculates, for every token, what is my worth in each frame
            

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h (t s) d -> (b t) s (h d)', t=F)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, qkv
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.temporal_attn_mask is not None:
            attn[self.temporal_attn_mask.expand_as(attn)] = -1e9

        if pad_mask is not None:
            attn[pad_mask.unsqueeze(1).expand_as(attn)] = -1e9

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        key = rearrange(key, '(b t) h n d -> b t n (h d)', t=self.num_frames)
        feats = key @ key.transpose(-1, -2)
        feats = feats > 0.2
        feats = torch.where(feats.type(torch.cuda.FloatTensor) == 0, 1e-5, feats)
        d_i = torch.sum(feats, dim=-1)
        D = torch.diag_embed(d_i)
        _, eigenvectors = torch.lobpcg(A=D-feats, B=D, k=2, largest=False)
        eigenvec = eigenvectors[:, :, :, 1]
        avg = torch.mean(eigenvec, dim=-1).unsqueeze(-1)
        bipartition = torch.gt(eigenvec , 0)
        bipartition = torch.where(avg > 0,bipartition, torch.logical_not(bipartition)) 
        bipartition = bipartition.to(torch.float)
        eigenvec = torch.abs(torch.mul(eigenvec, bipartition))
        eigenvec[eigenvec == 0] = -1e9
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
                 num_heads=12, mlp_ratio=4., temporal_depth=1, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., num_frames = 16, norm_layer=nn.LayerNorm, temporal_embed_dim = 768, temporal_num_heads = 8, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.temporal_attn_mask = create_mask(size=128, block_size=16)
        # self.temporal_norm = nn.LayerNorm()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, temporal_attn_mask=None)
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        self.temporal_norm = norm_layer(temporal_embed_dim)
        # Classifier head
        self.head = LinearClassifier(temporal_embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        return self.pos_drop(x)

    def forward(self, x, register_hook = False):
        B, C, T, H, W = x.shape
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, register_hook=register_hook)
            else:
                # return spatial_map of the last block
                x =  blk(x, register_hook=register_hook)

        x = rearrange(x, '(b t) n d -> b (t n) d', t=self.num_frames)
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