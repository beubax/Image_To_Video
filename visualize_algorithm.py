import os
import pickle
import torch
from torchvision import transforms as t
import torch.nn as nn
import time
# import torchshow as ts
from torch.utils.data import DataLoader
from dataset.hmdb51 import HMDB51
from dataset.utils import find_classes
from visualizations import visualize_eigvec, visualize_heatmap, visualize_heatmap2
# from visualizations import visualize_heatmap, visualize_point_cloud
from vit.utils import load_pretrained_weights
# from vit.vision_transformer_sparse import vit_base
from vit.vision_transformer_graph import vit_base, vit_small
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize, Permute, RandAugment, AugMix
from torchvision.transforms import transforms as T
# from torchvision.datasets import Kinetics
from torchvision.io import write_jpeg
from torchvision.utils import flow_to_image
from einops import rearrange
from torchvision.models.optical_flow import raft_large, raft_small


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = T.Compose(
    [
        ToTensorVideo(),  # C, T, H, W
        # Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
        # RandAugment(magnitude=0, num_layers=2),
        # Permute(dims=[1, 0, 2, 3]),  # C, T, H, W
        T.Resize(size=(224,224)),
        Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

train_metadata_file = "hmdb51-train-meta.pickle"
train_precomputed_metadata = None
if os.path.exists(train_metadata_file):
    with open(train_metadata_file, "rb") as f:
        train_precomputed_metadata = pickle.load(f)

train_set = HMDB51(
root="hmdb51",
annotation_path="annotations",
_precomputed_metadata=train_precomputed_metadata,
frames_per_clip=16,
step_between_clips=8,
frame_sample_rate=2,
train=True,
output_format="THWC",
transform=train_transform,
)

train_dataloader = DataLoader(
    train_set,
    batch_size=1,
    num_workers=4,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

class Spatial_Weighting(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv1, qkv2):
        query1 = qkv1[0]
        query2 = qkv2[0]
        key1 = qkv1[1]
        key2 = qkv2[1]
        key1 = rearrange(key1, '(b t) h n d -> b t n (h d)', t=self.num_frames)
        key2 = rearrange(key2, '(b t) h n d -> b t n (h d)', t=self.num_frames)
        query1 = rearrange(query1, '(b t) h n d -> b t n (h d)', t=self.num_frames)
        query2 = rearrange(query2, '(b t) h n d -> b t n (h d)', t=self.num_frames)
        feats1 = key1 @ query1.transpose(-1, -2)
        feats2 = key2 @ query2.transpose(-1, -2)
        # feats1 = nn.functional.normalize(feats1, p=2, dim=(-1,-2))
        # feats2 = nn.functional.normalize(feats2, p=2, dim=(-1,-2))
        print(feats1.mean(dim=-1).mean(dim=-1))
        print(feats2.mean(dim=-1).mean(dim=-1))
        feats = feats1
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
        eigenvec[eigenvec == 0] = float("-inf")
        eigenvec = self.softmax(eigenvec)
        return eigenvec
    
if not os.path.exists(train_metadata_file):
    with open(train_metadata_file, "wb") as f:
        pickle.dump(train_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

model = vit_base(num_classes=51).to(torch.device('cuda:0'))
load_pretrained_weights(model, model_name="vit_base", patch_size=16)
model.eval()
# flow_model = raft_large(pretrained=True, progress=False).to(torch.device('cuda:0'))
# flow_model = flow_model.eval()

# for batch in train_dataloader:
#     video, label = batch
#     output = model(video.to(torch.device('cuda:0')))
    # visualize_heatmap(spatial_map, video)
    # flow_video = []
    # print(video.shape)
    # video = video.permute(2, 0, 1, 3, 4)

    # for i, (img1, img2) in enumerate(zip(video, video[1:])):
    #     list_of_flows = flow_model(img1.to(torch.device('cuda:0')), img2.to(torch.device('cuda:0')))
    #     predicted_flow = list_of_flows[-1][0]
    #     flow_img = flow_to_image(predicted_flow)
    #     flow_video.append(flow_img.permute(1, 2, 0))
    #     output_folder = "output2/"  # Update this to the folder of your choice
    #     write_jpeg(flow_img.to("cpu"), output_folder + f"predicted_flow_{i}.jpg")

    # video = video.permute(1, 2, 0, 3, 4)
    # flow_video.append(flow_video[-1])
    # flow_video = torch.stack(flow_video)
    # flow_video = train_transform(flow_video)
    # print(time.time() - start)

    # output, qkv1 = model(video.to(torch.device('cuda:0')))
    # output, qkv2 = model(flow_video.unsqueeze(0))
    # spatial_weighting = Spatial_Weighting(num_frames=16)
    # spatial_map = spatial_weighting(qkv1, qkv2)
    # visualize_heatmap(spatial_map, video)
    # labels, class_to_idx = find_classes("hmdb51")
    # print(labels[label.item()])
    # break

for name, param in model.named_parameters():
   print('{}: {}'.format(name, param.requires_grad))
num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_total_param = sum(p.numel() for p in model.parameters())
print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    
