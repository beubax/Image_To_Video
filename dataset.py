import torch
import torchvision
from torchvision.datasets.folder import make_dataset
import itertools
import os
import random

def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_skip=4, frame_transform=None, video_transform=None, clip_len=16):
        super(VideoDataset).__init__()

        self.samples = get_samples(root)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.frame_skip = frame_skip

    def __iter__(self):
        for i in range(self.epoch_size):
            skip = self.frame_skip
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            while metadata["video"]['duration'][0] < (skip * (self.clip_len / metadata["video"]['fps'][0])):
                skip //= 2
            
            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (skip * (self.clip_len / metadata["video"]['fps'][0]))

            start = random.uniform(0., max_seek)
            for frame in itertools.islice(vid.seek(start), 0, skip * self.clip_len, skip):
                video_frames.append(self.frame_transform(frame['data'].float() / 255.0))
                current_pts = frame['pts']

            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            video = video.permute(1, 0, 2, 3)
            if self.video_transform:
                video = self.video_transform(video)
            
            yield video, target