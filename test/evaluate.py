"""
@Fire
https://github.com/fire717

This script is meant to run the model against the test set of a dataset.
It returns the predictions of the model and the ground truth values.
After this script, it is a best practice to also run the inspection script to visually inspect the predictions.

Eg: python -m evaluate --ckpt checkpoints/mpii_pretrained_h36m_finetuned/mpii_pretrained_h36m_finetuned.pth --dataset h36m
"""
import os, argparse
import random

from lib import init, Data, MoveNet, Task

from config import cfg
from lib.utils.utils import arg_parser


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

    data = Data(cfg)
    data_loader = data.getEvalDataloader()

    run_task = Task(cfg, model)

    run_task.modelLoad(cfg["ckpt"])

    run_task.evaluate(data_loader, fastmode=True)


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
