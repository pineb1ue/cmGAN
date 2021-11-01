import torchvision.transforms as T


def get_transforms(phase: str):
    """
    Image transformations

    Args:
        phase (str): 'train' or 'test'
        is_data_augmentation (bool): True or False
    Returns:
        T.Compose: Compose transforms
    """
    if phase not in ("train", "test"):
        raise ValueError

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms = []
    if phase == "train":
        transforms = [
            T.Resize((224, 224)),
            T.RandomRotation((0, 360)),
        ]
    elif phase == "test":
        transforms = [T.Resize((224, 224))]

    transforms += [T.ToTensor(), T.Normalize(mean, std)]

    return T.Compose(transforms)
