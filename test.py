import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import List

import torch
from data import infer_features_from_dir, Dataset
from models import FeatureGenerator


def my_evaluate(
    query_label: str,
    db_labels: np.ndarray,
    metrics: np.ndarray,
):
    db_labels = db_labels[np.argsort(metrics)]
    _, index = np.unique(db_labels, return_index=True)
    db_labels_unique = db_labels[np.sort(index)]

    acc = torch.IntTensor(len(db_labels_unique)).zero_()
    good_index = np.argwhere(db_labels_unique == query_label)
    good_index = good_index.flatten()
    acc[good_index[0] :] = 1

    return acc


def evaluate(
    query_label: str,
    db_labels: np.ndarray,
    metrics: np.ndarray,
    k: int,
):
    index = np.argsort(metrics)
    good_index = np.argwhere(db_labels == query_label)

    ap_tmp, cmc_tmp = compute_mAP(index, good_index, k)
    return ap_tmp, cmc_tmp


def compute_mAP(index: np.ndarray, good_index: np.ndarray, k: int):

    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # find good_index index
    n_k = min(k, len(good_index))
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0] :] = 1
    for i in range(n_k):
        d_recall = 1.0 / n_k
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap += d_recall * (old_precision + precision) / 2.0

    return ap, cmc


def save_json(
    query_features: np.ndarray,
    query_paths: List[str],
    db_features: np.ndarray,
    db_paths: List[str],
    save_path: str,
):

    # Initialize
    res_json = {}

    for query_feature, query_path in tqdm(
        zip(query_features, query_paths), total=len(query_paths)
    ):
        # Calculate euclidean distance
        metrics = np.array(
            [np.linalg.norm(query_feature - db_feature) for db_feature in db_features]
        )
        index_list = np.argsort(metrics)
        res_json[query_path] = [db_paths[idx] for idx in index_list][:10]

    with open(save_path, "w") as f:
        json.dump(res_json, f, indent=4)


def main(args):

    # Load DB features from pickle file
    with open(args.db_path, "rb") as f:
        db_dataset = pickle.load(f)
        db_labels = db_dataset["labels"]
        db_features = db_dataset["features"]
        db_paths = db_dataset["paths"]

    query_labels, query_features, query_paths = infer_features_from_dir(
        Dataset=Dataset,
        root_dir=args.query_dir,
        feature_generator=FeatureGenerator(),
        weight_path=args.weight_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize CMC and AP
    db_labels = np.array(db_labels)
    query_labels_unique = sorted(list(set(query_labels)))

    num_cows = {query_label: 0 for query_label in query_labels_unique}
    cmc = {
        query_label: torch.IntTensor(db_labels.size).zero_()
        for query_label in query_labels_unique
    }
    ap = {query_label: 0.0 for query_label in query_labels_unique}
    acc = {
        query_label: torch.IntTensor(np.unique(db_labels).size).zero_()
        for query_label in query_labels_unique
    }

    # Compute CMC and AP
    for i in tqdm(range(len(query_labels))):

        metrics = np.array(
            [
                np.linalg.norm(query_features[i] - db_feature)
                for db_feature in db_features
            ]
        )

        ap_tmp, cmc_tmp = evaluate(query_labels[i], db_labels, metrics, k=args.k)
        acc_tmp = my_evaluate(query_labels[i], db_labels, metrics)

        if cmc_tmp[0] == -1:
            continue

        cmc[query_labels[i]] += cmc_tmp
        ap[query_labels[i]] += ap_tmp
        acc[query_labels[i]] += acc_tmp

        # Get number of each query
        num_cows[query_labels[i]] += 1

    # Compute average CMC and mAP
    average_cmc = 0.0
    mAP = 0.0
    average_acc = 0.0

    for query_label in query_labels_unique:
        average_cmc += cmc[query_label].float() / num_cows[query_label]
        mAP += ap[query_label] / num_cows[query_label]
        average_acc += acc[query_label].float() / num_cows[query_label]

    average_cmc /= len(query_labels_unique)
    mAP /= len(query_labels_unique)
    average_acc /= len(query_labels_unique)

    average_cmc = [round(float(cmc_k) * 100, 1) for cmc_k in average_cmc]
    mAP = round(mAP * 100, 1)
    average_acc = [round(float(acc_k) * 100, 1) for acc_k in average_acc]

    print(
        f"CMC: Rank@1:{average_cmc[0]}% Rank@10:{average_cmc[9]}% Rank@20:{average_cmc[19]}%"
    )
    print(f"mAP@{args.k}: {mAP}%")
    print(
        f"My ACC: Top@1:{average_acc[0]}% Top@3:{average_acc[2]}% Top@5:{average_acc[4]}%"
    )

    # Save result as json
    if args.json:
        save_json(
            query_features,
            query_paths,
            db_features,
            db_paths,
            save_path=args.db_path.split("/")[-1].split(".")[0],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-db", "--db-path", type=str, help="dastabase path (pickle file)"
    )
    parser.add_argument("-q", "--query-dir", type=str, help="input query dir path")

    parser.add_argument("-w", "--weight-path", type=str, help="input weight path")
    parser.add_argument("-bs", "--batch-size", type=int, default=32)
    parser.add_argument("-nw", "--num-workers", type=int, default=2)
    parser.add_argument("-j", "--json", action="store_true")
    parser.add_argument("-k", "--k", type=int, default=10)
    args = parser.parse_args()

    main(args)
