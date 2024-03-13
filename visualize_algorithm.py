import torch
from torchvision import transforms as t
import torchshow as ts
from torch.utils.data import DataLoader
from dataset import VideoDataset
from visualizations import visualize_heatmap
from vit.utils import load_pretrained_weights
from vit.vision_transformer import vit_base


transforms = [t.Resize((224, 224)), t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
frame_transform = t.Compose(transforms)

dataset = VideoDataset("./Videos", epoch_size=None, frame_transform=frame_transform)
loader = DataLoader(dataset, batch_size=1)
model = vit_base(num_classes=51).cuda()
load_pretrained_weights(model, model_name="vit_base", patch_size=16)
model.eval()
for batch in loader:
    video = batch["video"].to(torch.device('cuda:0'))
    print(video.shape)
    output, spatial_attention_map = model(video)
    visualize_heatmap(spatial_attention_map, batch["video"], scales=[16,16], dims=(14, 14))
    print(output.shape)
    break
