import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_transforms
from models import FeatureGenerator, IdClassifier, DomainClassifier
from data import TripletDataset
from engine import TripletTrainer
from utils import seed_worker


SEED = 42


def train(args):

    # Set random seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    # Load data
    train_dataset = TripletDataset(
        root_dir_3d=args.train_dir_3d,
        root_dir_2d=args.train_dir_2d,
        transforms=get_transforms("train"),
    )

    # Load dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # Get model
    feature_generator = FeatureGenerator()
    id_classifier = IdClassifier(num_classes=train_dataset.num_classes)
    domain_classifier = DomainClassifier()

    # Setting Loss
    criterion_triplet = nn.TripletMarginLoss(margin=1.4)
    criterion_id = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    # Setting other params
    optimizer = torch.optim.SGD(
        [
            {"params": feature_generator.parameters(), "lr": 1e-4},
            {"params": id_classifier.parameters(), "lr": 1e-4},
            {"params": domain_classifier.parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = TripletTrainer(
        train_loader=train_loader,
        feature_generator=feature_generator,
        id_classifier=id_classifier,
        domain_classifier=domain_classifier,
        criterion_triplet=criterion_triplet,
        criterion_id=criterion_id,
        criterion_domain=criterion_domain,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        device=device,
        interval=args.interval,
        output_dir=os.path.join("weights", args.output_dir),
        project_name=os.path.join("runs", args.output_dir),
    )
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t3d", "--train-dir-3d", type=str)
    parser.add_argument("-t2d", "--train-dir-2d", type=str)
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("-m", "--metric", default="triplet")

    parser.add_argument("-w", "--weights-path")

    parser.add_argument("-bs", "--batch-size", type=int, default=128)
    parser.add_argument("-nw", "--num-workers", type=int, default=2)
    parser.add_argument("-ne", "--num-epochs", type=int, default=100)
    parser.add_argument("-it", "--interval", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(os.path.join("weights", args.output_dir), exist_ok=True)
    train(args)
