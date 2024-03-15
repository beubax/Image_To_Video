import click
import os
import pickle
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from dataset.hmdb51 import HMDB51
from lightning_module import VideoLightningModule


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-nc", "--num-classes", type=int, default=51, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=32, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=16, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option("--max-epochs", type=int, default=50, help="max epochs.")
@click.option("--num-workers", type=int, default=4)
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option("--preview-video", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
def main(
    dataset_root,
    annotation_path,
    num_classes,
    batch_size,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    fast_dev_run,
    seed,
    preview_video,
):
    
    pl.seed_everything(seed)

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

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    
    train_metadata_file = "hmdb51-train-meta.pickle"
    train_precomputed_metadata = None
    if os.path.exists(train_metadata_file):
        with open(train_metadata_file, "rb") as f:
            train_precomputed_metadata = pickle.load(f)

    train_set = HMDB51(
    root=dataset_root,
    annotation_path=annotation_path,
    _precomputed_metadata=train_precomputed_metadata,
    frames_per_clip=frames_per_clip,
    step_between_clips=8,
    frame_sample_rate=4,
    train=True,
    output_format="THWC",
    transform=train_transform,
)

    if not os.path.exists(train_metadata_file):
        with open(train_metadata_file, "wb") as f:
            pickle.dump(train_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    val_metadata_file = "hmdb51-val-meta.pickle"
    val_precomputed_metadata = None
    if os.path.exists(val_metadata_file):
        with open(val_metadata_file, "rb") as f:
            val_precomputed_metadata = pickle.load(f)

    val_set = HMDB51(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=val_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        step_between_clips=8,
        frame_sample_rate=4,
        train=False,
        output_format="THWC",
        transform=test_transform,
    )

    if not os.path.exists(val_metadata_file):
        with open(val_metadata_file, "wb") as f:
            pickle.dump(val_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    model = VideoLightningModule(
        num_classes=num_classes,
        lr=3e-4,
        weight_decay=0.001,
        max_epochs=max_epochs,
    )

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    logger = TensorBoardLogger("logs", name="VVIT")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint("./vvit_hdmb51.ckpt")


if __name__ == "__main__":
    main()
