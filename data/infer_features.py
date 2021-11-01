import numpy as np
from tqdm import tqdm
from typing import Tuple, List

import torch
import torchvision

from .dataset import QueryDataset
from .transforms import get_transforms


def infer_features_from_dir(
    Dataset: torch.utils.data.Dataset,
    root_dir: str,
    feature_generator: torchvision.models,
    weight_path: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[str], np.ndarray]:
    dataset = Dataset(root_dirs=root_dir, transforms=get_transforms("test"))

    labels = [data["path"].split("/")[-2] for data in dataset.data]
    features = _infer_features(
        dataset, feature_generator, weight_path, batch_size, num_workers
    )
    paths = [data["path"] for data in dataset.data]

    return labels, features, paths


def infer_features_from_imgs(
    imgs: np.ndarray,
    feature_generator: torchvision.models,
    weight_path: str,
    batch_size: int,
    num_workers: int,
):
    dataset = QueryDataset(imgs=imgs, transforms=get_transforms("test"))
    features = _infer_features(
        dataset, feature_generator, weight_path, batch_size, num_workers
    )
    paths = [data["path"] for data in dataset.data]
    return features, paths


def _infer_features(
    dataset: torch.utils.data.Dataset,
    feature_generator: torchvision.models,
    weight_path: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, List[str]]:

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_generator.load_state_dict(torch.load(weight_path))
    feature_generator.to(device)
    feature_generator.eval()

    # Embeddings/labels to be stored on the testing set
    features = np.zeros((1, 2048))

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    for images, _ in tqdm(data_loader):
        # Put the model on the GPU and in evaluation mode
        images = images.to(device)
        # Get the embeddings of this batch of images
        outputs = feature_generator(images)
        # Express embeddings in numpy form
        embeddings = outputs.data
        embeddings = embeddings.cpu().numpy()
        # Store testing data on this batch ready to be evaluated
        features = np.concatenate((features, embeddings), axis=0)

    # Remove zeros
    features = np.delete(features, obj=0, axis=0)

    return features
