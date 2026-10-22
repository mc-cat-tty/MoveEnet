"""
Author: @mc-cat-tty

Visually inspect the outputs (aka the predictions) of the evaluation script.

Eg: python -m test.inspect --evaluations-path ../datasets/eros_sequential/h36m/ --images-path ../datasets/h36m/test/h36m_EROS/ --gt-labels-json ../datasets/h36m/test/h36m_anno/gt_poses.json --dataset-name h36m
Tip: when the dashboard opens, settings->render frames as video can be very useful to check the correctness of the predictions and the GT
"""

from __future__ import annotations
import argparse, json
import fiftyone as fo
import pandas as pd
from pathlib import Path
from lib.visualization.visualization import hpecore_kps_labels, lines_in_skeleton, MOVENET_TO_DHP19_INDICES
from dataclasses import dataclass
from typing import Iterable


MOVENET_TO_HPECORE_ORDER = MOVENET_TO_DHP19_INDICES[:, 0].tolist()
JOINTS_COORDINATES_COLUMNS = [
    f"{joint}_{axis}" for joint in hpecore_kps_labels.keys() for axis in ("x", "y")
]
X_COLUMNS = [c for c in JOINTS_COORDINATES_COLUMNS if c.endswith("_x")]
Y_COLUMNS = [c for c in JOINTS_COORDINATES_COLUMNS if c.endswith("_y")]
PREDICTION_COLUMNS = ["ts", "delay"] + JOINTS_COORDINATES_COLUMNS

### Ground Truth data structures and helpers ###
@dataclass
class GroundTruthJSONLine:
    img_name: str
    ts: float
    keypoints: list[float]
    original_sample: str
    center: tuple[float, float] | None = None
    torso_size: float | None = None
    keypoints_prev: list[float] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> GroundTruthJSONLine:
        return cls(**d)


class GroundTruthIndexedByImgName(dict):
    @classmethod
    def from_gt_json_lines(cls, gt_lines: Iterable[GroundTruthJSONLine]) -> GroundTruthIndexedByImgName:
        return cls({
            gt_json_line.img_name: gt_json_line for gt_json_line in gt_lines
        })

    @classmethod
    def from_dicts(cls, dicts: Iterable[dict]) -> GroundTruthIndexedByImgName:
        return cls.from_gt_json_lines(
            map(GroundTruthJSONLine.from_dict, dicts)
        )

def get_keypoints_from_json_line(line: GroundTruthJSONLine) -> fo.Keypoints:
    # Each keypoint triple is (x, y, visibility)
    kpt_triples = [line.keypoints[i:i + 3] for i in range(0, len(line.keypoints), 3)]

    # The GT JSON follows the MoveNet joint order, the skeleton the HPECore one
    return fo.Keypoints(
        keypoints = [
            fo.Keypoint(
                points = [kpt_triples[i][:2] for i in MOVENET_TO_HPECORE_ORDER]
            )
        ]
    )

### Predictions data structures and helpers ###
def get_predictions_df_from_dir(dir: Path, sequences: Iterable[str], img_w: int, img_h: int) -> pd.DataFrame:
    """
    Returns a dataframe indexed by sequence name, timestamp. Each row contains the fields defined by PREDICTION_COLUMNS
    Peforms coordinates normalization.
    """
    sequences = set(sequences)
    csv_dfs = []

    for seq in sequences:
        # v TODO: make more robust, atm we assume just one csv is in the folder
        csv_filenames = tuple((dir / seq).glob("*.csv"))
        assert len(csv_filenames) > 0, f"Found zero CSV files in {dir / seq}... Aborting"
        assert len(csv_filenames) == 1, f"Found more than one CSV files in {dir / seq}... Aborting"
        csv_filename = csv_filenames[0]

        csv_df = pd.read_csv(csv_filename, sep=" ", header=None, names=PREDICTION_COLUMNS)
        csv_df["sequence"] = seq
        csv_df["ts"] = csv_df["ts"].round(2)
        csv_df[X_COLUMNS] /= img_w
        csv_df[Y_COLUMNS] /= img_h
        csv_dfs.append(csv_df)
    
    predictions_df = pd.concat(csv_dfs, ignore_index=True).set_index(["sequence", "ts"])
    return predictions_df


def get_keypoints_from_predictions_df(
    predictions_df: pd.DataFrame, sequences: Iterable[str], timestamps: Iterable[float]
) -> list[fo.Keypoints]:
    """
    Performs conversion to Fiftyone Keypoints, one per (sequence, timestamp) pair, in the given order.
    """
    keys = pd.MultiIndex.from_arrays([sequences, [round(ts, 2) for ts in timestamps]])
    rows = predictions_df.reindex(keys)

    missing = rows[JOINTS_COORDINATES_COLUMNS].isna().any(axis=1).to_numpy()
    assert not missing.any(), f"Missing predictions for {missing.sum()} samples, eg: {keys[missing][:3].tolist()}"

    # (n_samples, n_joints, 2); predictions are already in HPECore order, no reordering needed
    points = rows[JOINTS_COORDINATES_COLUMNS].to_numpy().reshape(len(rows), -1, 2)

    return [
        fo.Keypoints(keypoints = [fo.Keypoint(points = p)])
        for p in points.tolist()
    ]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="The dataset name is used to build an entry to the Fiftyone database and cache the dataset content."
    )    
    parser.add_argument(
        "--evaluations-path", type=Path, required=True,
        help="Path to the folder containing the output of the test set evaluations. Expected structure is <evaluations-path>/<seq-name>/<csv-labels>"
    )
    
    parser.add_argument(
        "--images-path", type=Path, required=True,
        help="Path to the folder containing the images used by the test set evaluation script. Expected structure is <images-path>/<jpeg-images>"
    )
    
    parser.add_argument(
        "--gt-labels-json", type=Path, required=True,
        help="Path to the JSON file containing the ground truth labels; this file provides the link between input images, timestamps, and output predictions. Expected structure is <gt-labels-path>/<json-file>"
    )
    
    parser.add_argument(
        "--parallel-comparisons", default=1, type=int,
        help="Number of parallel comparisons open in the inspection window"
    )
    
    parser.add_argument(
        "-f", "--force-rebuild", action='store_true', default=False,
        help="Forcibly rebuild the Dataset in Fiftyone database",
    )

    parser.add_argument(
        "--dev-run", action='store_true', default=False,
        help="Development run; only 10 images are loaded from the disk",
    )

    
    return parser.parse_args()


def build_dataset(args) -> fo.Dataset:
    hpecore_skeleton = fo.KeypointSkeleton(
        labels = hpecore_kps_labels.keys(),
        edges = [
            (hpecore_kps_labels[kpt_start_name], hpecore_kps_labels[kpt_end_name])
            for (kpt_start_name, kpt_end_name) in lines_in_skeleton
        ]
    )

    with open(args.gt_labels_json, "r") as gt_file:
        ground_truth_json = json.load(gt_file)
    ground_truth_poses = GroundTruthIndexedByImgName.from_dicts(ground_truth_json)
    
    if args.dev_run:
        image_paths = sorted(args.images_path.glob("*.jpg"))[:10]
        dataset = fo.Dataset.from_images([str(p) for p in image_paths], name=args.dataset_name)
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir = args.images_path,
            dataset_type = fo.types.ImageDirectory, 
            name = args.dataset_name,
            persistent = True
        )
    
    dataset.default_skeleton = hpecore_skeleton
    dataset.compute_metadata()
    # v TODO: make more robust, we assume all images are same w, h
    img_w, img_h = dataset.first().metadata.width, dataset.first().metadata.height

    filepaths = dataset.values("filepath")
    
    if not filepaths:
        print("filepaths not availabe in dataset")
        exit(1)

    ground_truths = [
        get_keypoints_from_json_line(ground_truth_poses[Path(fp).name]) for fp in filepaths
    ]

    sequences = [
        ground_truth_poses[Path(fp).name].original_sample for fp in filepaths
    ]

    timestamps = [
        ground_truth_poses[Path(fp).name].ts for fp in filepaths
    ]

    predictions_df = get_predictions_df_from_dir(args.evaluations_path, sequences, img_w, img_h)
    predictions = get_keypoints_from_predictions_df(predictions_df, sequences, timestamps)
    
    dataset.set_values("sequence", sequences)
    dataset.set_values("timestamp", timestamps)
    dataset.set_values("ground_truth", ground_truths)
    dataset.set_values("prediction", predictions)

    return dataset


def main():
    args = parse_args()

    dataset_exists = fo.dataset_exists(args.dataset_name)

    if dataset_exists and args.force_rebuild:
        fo.delete_dataset(args.dataset_name)
        dataset_exists = False

    if dataset_exists:
        dataset = fo.load_dataset(args.dataset_name)
    else:
        dataset = build_dataset(args)

    session = fo.launch_app(dataset, address="0.0.0.0")
    
    view = dataset.group_by("sequence", order_by="timestamp")
    session.view = view

    session.show()
    session.wait()

if __name__ == "__main__":
    main()