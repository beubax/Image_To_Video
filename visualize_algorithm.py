import os
import pickle
import torch
from torchvision import transforms as t
# import torchshow as ts
from torch.utils.data import DataLoader
from dataset.hmdb51 import HMDB51
# from visualizations import visualize_heatmap, visualize_point_cloud
from vit.utils import load_pretrained_weights
# from vit.vision_transformer_sparse import vit_base
from vit.vision_transformer_sparse import vit_base
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize, Permute, RandAugment, AugMix
from torchvision.transforms import transforms as T
# from torchvision.datasets import Kinetics


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = T.Compose(
    [
        ToTensorVideo(),  # C, T, H, W
        Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
        RandAugment(magnitude=0, num_layers=2),
        Permute(dims=[1, 0, 2, 3]),  # C, T, H, W
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

if not os.path.exists(train_metadata_file):
    with open(train_metadata_file, "wb") as f:
        pickle.dump(train_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

model = vit_base(num_classes=51).to(torch.device('cuda:0'))
load_pretrained_weights(model, model_name="vit_base", patch_size=16)
model.eval()

for batch in train_dataloader:
    video, label = batch
    # print(label)
    output, indices = model(video.to(torch.device('cuda:0')))
    break
#     ts.show_video(video[0].permute(1, 0, 2, 3))
#     visualize_point_cloud(video, indices)
#     # print(output.shape)
#     break

# for name, param in model.named_parameters():
#    print('{}: {}'.format(name, param.requires_grad))
# num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
# num_total_param = sum(p.numel() for p in model.parameters())
# print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    
