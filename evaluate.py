import os
import pickle

import click
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import torch
import seaborn as sns
from pytorchvideo.transforms import Normalize
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
from dataset.utils import find_classes
from dataset.hmdb51 import HMDB51
from lightning_module import VideoLightningModule


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-m", "--model-path", type=click.Path(exists=True), required=True, help="path to model weight.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-nc", "--num-classes", type=int, default=51, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=4, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=16, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option("--num-workers", type=int, default=4)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option("--verbose", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
def main(
    dataset_root,
    model_path,
    annotation_path,
    num_classes,
    batch_size,
    frames_per_clip,
    video_size,
    num_workers,
    seed,
    verbose,
):
    pl.seed_everything(seed)

    labels, _ = find_classes(dataset_root)

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_metadata_file = "hmdb51-train-meta.pickle"
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
        frame_sample_rate=2,
        train=True,
        output_format="THWC",
        transform=test_transform,
    )

    if not os.path.exists(val_metadata_file):
        with open(val_metadata_file, "wb") as f:
            pickle.dump(val_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )

    model = VideoLightningModule.load_from_checkpoint(model_path)
    trainer = pl.Trainer(accelerator="auto", default_root_dir="lightning_predict_logs")
    predictions = trainer.predict(model, dataloaders=val_dataloader)

    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])

    print("accuracy:", accuracy(y_prob, y, task="multiclass", num_classes=num_classes))
    print("accuracy_top5:", accuracy(y_prob, y, task="multiclass", num_classes=num_classes, top_k=5))
    print("auroc:", auroc(y_prob, y, task="multiclass", num_classes=num_classes))
    print("f1_score:", f1_score(y_prob, y, task="multiclass", num_classes=num_classes))

    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)

    plt.figure(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("output.png", dpi=300)
    if verbose:
        plt.show()


if __name__ == "__main__":
    main()
