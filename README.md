# MoveEnet

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/event-driven-robotics/MoveEnet/blob/main/LICENSE)

## Intro
![start](/data/imgs/moveEnet.gif)

MoveEnet is an online event-driven Human Pose Estimation model. It is based on MoveNet architecture from google and since google did not release a training code, this repository was originally forked from [movenet.pytorch](https://github.com/fire717/movenet.pytorch) repository.

If you use MoveEnet for your scientific publication, please cite:

```
@InProceedings{Goyal_2023_CVPR,
    author    = {Goyal, Gaurvi and Di Pietro, Franco and Carissimi, Nicolo and Glover, Arren and Bartolozzi, Chiara},
    title     = {MoveEnet: Online High-Frequency Human Pose Estimation With an Event Camera},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {4023-4032}
}
```
Article PDF file: [MoveEnet: Online High-Frequency Human Pose Estimation With an Event Camera](https://github.com/user-attachments/files/17659249/MoveEnet-CVPR-WEBV2023.pdf)

The eH3.6m dataset is available here:
```
https://zenodo.org/record/7842598
```
The primary python libraries needed are listed in ``requirements.txt``.

## How To Run

1. Download the [hpe-core](https://github.com/event-driven-robotics/hpe-core/) repository. Follow the steps laid out to export the training data for the [eh36m dataset](https://github.com/event-driven-robotics/hpe-core/tree/main/datasets/h36m), run all the scripts described in *datasets/h36m/README.md*. This will lead to creation of a folder of images, and an annotation file called ``pose.json``.

2. Separate the annotation file to training and val. Ensure the path to the ``pose.json`` is added to ``scripts/data/split_trainval.py``. Changes can be made in this file for specific separation of data between training and validations splits. Then run the file using the following command:

```
python3 scripts/data/split_trainval.py
```

3. The resulting data should be arranged in the following format.

```
├── data
    ├── annotations (train.json, val.json)
    └── eros   (xx.jpg, xx.jpg,...)

```



```
KEYPOINTS_MAP = {'head': 0, 'shoulder_right': 1, 'shoulder_left': 2, 'elbow_right': 3, 'elbow_left': 4,
                     'hip_left': 5, 'hip_right': 6, 'wrist_right': 7, 'wrist_left': 8, 'knee_right': 9, 'knee_left': 10,
                     'ankle_right': 11, 'ankle_left': 12}
```

4. For more on the data format, you can explore the dataloader file lib/data/data_tools.py and the class 'TensorDataset'. 

5. Open the config.py file and set the following path keys to the correct paths for your dataset:
```
    cfg["img_path"]=<relevant-path>/data/eros
    cfg["train_label_path"]=<relevant-path>/data/annotations/train.json
    cfg["val_label_path"]=<relevant-path>/data/annotations/val.json
```
6. In case you wish to start the training form a specific checkpoint, use the following key in the config file:
```
    cfg["ckpt"]=/path/to/model.ckpt
```
7. To train a network:
```
python train.py
```

8. To predict results using a specific checkpoint, use files `predict.py` and `evaluate.py`. To create a video from an input data, use the file `create_video.py`.

# Notes
- Any  contribution to make the code more usable would be appreciated. Feel free to fork and create merge request to contribute back.
- For inference-only code, or for inference on camera directly, follow the directions on the [hpe-core repository](https://github.com/event-driven-robotics/hpe-core/tree/main/example/movenet).