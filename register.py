import pickle
import argparse

from data import infer_features_from_dir, Dataset
from models import FeatureGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", "--db-dir", type=str, help="input db dir path")
    parser.add_argument("-en", "--embedding-net", type=str, default="resnet50")
    parser.add_argument("-w", "--weight-path", type=str, help="weight path")
    parser.add_argument("-bs", "--batch-size", type=int, default=128)
    parser.add_argument("-nw", "--num-workers", type=int, default=2)
    parser.add_argument("-sp", "--save-path", type=str, help="save path (hoge.pkl)")

    args = parser.parse_args()

    db_labels, db_features, db_paths = infer_features_from_dir(
        Dataset=Dataset,
        root_dir=args.db_dir,
        feature_generator=FeatureGenerator(),
        weight_path=args.weight_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    with open(args.save_path, "wb") as f:
        pickle.dump(
            {"labels": db_labels, "features": db_features, "paths": db_paths}, f
        )
