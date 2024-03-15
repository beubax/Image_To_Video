import os
import pickle
import torch
from torchvision import transforms as t
# import torchshow as ts
from torch.utils.data import DataLoader
from dataset.hmdb51 import HMDB51
from visualizations import visualize_heatmap
from vit.utils import load_pretrained_weights
from vit.vision_transformer import vit_base
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize, Permute, RandAugment, AugMix
from torchvision.transforms import transforms as T


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = T.Compose(
    [
        ToTensorVideo(),  # C, T, H, W
        Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
        RandAugment(magnitude=5, num_layers=2),
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
    annotation_path="testTrainMulti_7030_splits",
    _precomputed_metadata=train_precomputed_metadata,
    frames_per_clip=16,
    step_between_clips=8,
    frame_sample_rate=4,
    train=True,
    output_format="THWC",
    transform=train_transform,
)
if not os.path.exists(train_metadata_file):
    with open(train_metadata_file, "wb") as f:
        pickle.dump(train_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)


train_dataloader = DataLoader(
        train_set,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )


# model = vit_base(num_classes=51).to(torch.device('cuda:0'))
# load_pretrained_weights(model, model_name="vit_base", patch_size=16)
# model.eval()
for batch in train_dataloader:
    video, label = batch
    # print(label)
    # output, spatial_attention_map = model(video.to(torch.device('cuda:0')))
    # visualize_heatmap(spatial_attention_map, video, scales=[16,16], dims=(14, 14))
    # print(output.shape)
    # break
    
