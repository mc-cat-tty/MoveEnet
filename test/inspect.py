"""
Author: @mc-cat-tty

Visually inspect the outputs (aka the predictions) of the evaluation script.

Eg: python -m test.inspect --evaluations-path ../datasets/eros_sequential/h36m/ --images-path ../datasets/h36m/test/h36m_EROS/ --gt-labels-json ../datasets/h36m/test/h36m_anno/gt_poses.json --dataset-name h36m
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import fiftyone as fo
from lib.visualization.visualization import hpecore_kps_labels, lines_in_skeleton
from dataclasses import dataclass
from typing import Iterable


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

    return fo.Keypoints(
        keypoints = [
            fo.Keypoint(
                points = [[x, y] for x, y, _ in kpt_triples]
            )
        ]
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-path", help="Path to the folder containing the output of the test set evaluations. Expected structure is <evaluations-path>/<dataset-name>/<seq-name>/<csv-labels>", type=Path)
    parser.add_argument("--images-path", help="Path to the folder containing the images used by the test set evaluation script. Expected structure is <images-path>/<jpeg-images>", type=Path)
    parser.add_argument("--gt-labels-json", help="Path to the JSON file containing the ground truth labels; this file provides the link between input images, timestamps, and output predictions. Expected structure is <gt-labels-path>/<json-file>", type=Path)
    parser.add_argument("--parallel-comparisons", help="Number of parallel comparisons open in the inspection window", default=1, type=int)
    parser.add_argument("--dataset-name", help="Name of the dataset we are inspecting", default="h36m")
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

    dataset = fo.Dataset.from_dir(
        dataset_dir = args.images_path,
        dataset_type = fo.types.ImageDirectory, 
        name=args.dataset_name
    )
    dataset.default_skeleton = hpecore_skeleton


    filepaths = dataset.values("filepath")
    
    if not filepaths:
        print("filepaths not availabe in dataset")
        exit(1)

    ground_truths = (
        get_keypoints_from_json_line(ground_truth_poses[Path(fp).name]) for fp in filepaths
    )
    dataset.set_values("ground_truth", ground_truths)


def main():
    args = parse_args()

    if fo.dataset_exists(args.dataset_name):
        dataset = fo.load_dataset(args.dataset_name)
    else:
        dataset = build_dataset(args)
    
    session = fo.launch_app(dataset)
    session.show()
    session.wait()

if __name__ == "__main__":
    main()