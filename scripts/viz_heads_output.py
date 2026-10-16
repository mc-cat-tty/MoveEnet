"""
Author: @mc-cat-tty

Visualization script to create nice images of the four heads of MoveENet.
Input: a EROS-like image. The fist batch returned by the dataloader is used as input, the dataset path is read from the configuration file.
Output: person center, heatmap, keypoints regression, keypoints offsets, heatmap+regressions heatmap and superimposition pictures, stored
in the output folder specified by the --output-dir parameter.
"""

import cv2
import numpy as np
import torch, torchvision, argparse
import torch.nn.functional as F
from torchvision.utils import save_image
from lib import MoveNet
from config import cfg as default_cfg
from pathlib import Path
from lib.data.data import Data
from typing import Iterable

def assert_chw(t: torch.Tensor, name: str):
    assert t.ndim == 3, f"{name} must be a CHW tensor, got {t.ndim} dims (shape {tuple(t.shape)})"
    assert not t.is_cuda, f"{name} must be on the CPU"

def assert_img(img: torch.Tensor):
    assert_chw(img, "img")
    assert img.is_floating_point(), f"img must be a float tensor in [0, 1], not {img.dtype}"
    assert img.min() >= 0 and img.max() <= 1 + 1e-6, f"img values must be in [0, 1], got [{img.min()}, {img.max()}]"

def assert_even_channels(t: torch.Tensor, name: str):
    assert t.shape[0] % 2 == 0, f"{name} must have an even number of channels (x, y pairs), not {t.shape[0]}"

def draw_vector(img: torch.Tensor, start_xy, end_xy, color=(255, 255, 255), thickness=1) -> torch.Tensor:
    assert_img(img)
    canvas = np.ascontiguousarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))  # CHW [0,1] -> HWC uint8
    pt1 = (int(start_xy[0]), int(start_xy[1]))
    pt2 = (int(end_xy[0]), int(end_xy[1]))
    cv2.arrowedLine(canvas, pt1, pt2, color, thickness, tipLength=0.1)
    return torch.from_numpy(canvas).permute(2, 0, 1).float() / 255

def get_center_pixel_xy(center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center_y = center.max(dim=-1).values.argmax()
    center_x = center.max(dim=-2).values.argmax()
    return center_x, center_y

def get_raw_keypoints_xy(center: torch.Tensor, keypoints_regression: torch.Tensor) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    cx, cy = get_center_pixel_xy(center)
    keypoints_iterator = zip(keypoints_regression[0::2, cy, cx], keypoints_regression[1::2, cy, cx])
    return keypoints_iterator

def get_absolute_keypoints_xy(center: torch.Tensor, keypoints_regression: torch.Tensor) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    assert center.shape[-2:] == keypoints_regression.shape[-2:], f"Center and keypoints regression tensors must have the same (H, W) shape, not {center.shape[-2:]} and {keypoints_regression.shape[-2:]}"
    cx, cy = get_center_pixel_xy(center)
    keypoints = get_raw_keypoints_xy(center, keypoints_regression)
    for kpt_x, kpt_y in keypoints:
        kpt_x, kpt_y = cx + int(kpt_x+0.5), cy + int(kpt_y+0.5)  # computing absolute kpt position, rounding towards zero
        kpt_x = torch.min(torch.max(torch.tensor(0), kpt_x), torch.tensor(center.shape[-1]-1))  # clamping
        kpt_y = torch.min(torch.max(torch.tensor(0), kpt_y), torch.tensor(center.shape[-2]-1))
        yield kpt_x, kpt_y

def get_hm_reg_alignment_scores(center: torch.Tensor, keypoints_regression: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
    assert_chw(center, "center")
    assert_chw(keypoints_regression, "keypoints_regression")
    assert_chw(heatmap, "heatmap")
    assert_even_channels(keypoints_regression, "keypoints_regression")
    assert center.shape[0] == 1, f"Center tensor must have 1 channel, not {center.shape[0]}"
    assert keypoints_regression.shape[0] == heatmap.shape[0]*2, f"Keypoints regression must have twice the channels of the heatmap, instead of {keypoints_regression.shape[0]} and {heatmap.shape[0]}"
    assert center.shape[-2:] == keypoints_regression.shape[-2:] == heatmap.shape[-2:], "Center, regressions, and heatmap tensors (H, W) shape must be the same"
    
    C, H, W = heatmap.shape
    cx, cy = get_center_pixel_xy(center)

    r = keypoints_regression[:, cy, cx]
    rx = cx + (r[0::2] + 0.5).trunc()
    ry = cy + (r[1::2] + 0.5).trunc()

    ys = torch.arange(H).view(1, H, 1)
    xs = torch.arange(W).view(1, 1, W)

    scores = heatmap.clone()

    dist = torch.hypot(xs - rx.view(C, 1, 1), ys - ry.view(C, 1, 1))
    scores /= (dist + 1.8)

    return scores

def get_joint_locations_xy(scores: torch.Tensor) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a tuple of tensors: x coordinates fo the joints, y coordinates of the joints.
    """
    joints_y = scores.max(dim=-1).values.argmax(dim=-1)
    joints_x = scores.max(dim=-2).values.argmax(dim=-1)
    return zip(joints_x, joints_y)

def get_offsets_at_joint_locations_xy(scores: torch.Tensor, offsets: torch.Tensor) -> Iterable[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Yields a tuple that contains joint coordinates and offsets coordinates (joint_x, joint_y), (offset_x, offset_y)
    """
    for ch, (jx, jy) in enumerate(get_joint_locations_xy(scores)):
        ox, oy = offsets[2*ch, jy, jx], offsets[2*ch+1, jy, jx]
        yield (jx, jy), (ox, oy)

def superimpose_heatmap(img: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    assert_img(img)
    assert_chw(heatmap, "heatmap")
    assert heatmap.min() >= 0 and heatmap.max() <= 1, f"heatmap values must be in [0, 1], got [{heatmap.min()}, {heatmap.max()}]"

    hm_one_channel = heatmap.amax(0, keepdim=True)
    hm_resized = F.interpolate(hm_one_channel.unsqueeze(0), size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=False)[0, 0]
    hm_u8 = (hm_resized*255).byte().numpy()
    hm_colormapped = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)  # BGR uint8 (H, W, 3)
    hm_colormapped = cv2.cvtColor(hm_colormapped, cv2.COLOR_BGR2RGB)
    hm_colormapped = torch.from_numpy(hm_colormapped).permute(2, 0, 1).float() / 255
    blending_mask = (hm_resized * alpha).unsqueeze(0)
    return (1-blending_mask) * img + blending_mask * hm_colormapped

def superimpose_center_dot(img: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    assert_img(img)
    assert_chw(center, "center")
    assert center.shape[0] == 1, f"Center tensor must have 1 channel, not {center.shape[0]}"

    center_x, center_y = get_center_pixel_xy(center)
    H, W = img.shape[-2:]
    center_dot_overlay = torch.zeros((H, W), dtype=torch.bool)
    scale_factor_y = img.shape[-2] / center.shape[-2]
    scale_factor_x = img.shape[-1] / center.shape[-1]
    center_y = int(center_y * scale_factor_y)
    center_x = int(center_x * scale_factor_x)
    center_dot_overlay[
        max(center_y-2, 0):min(center_y+2, H),
        max(center_x-2, 0):min(center_x+2, W)] = True
    return center_dot_overlay

def superimpose_center(img: torch.Tensor, center: torch.Tensor, alpha: float = 0.5, add_dot=False) -> torch.Tensor:
    """
    Superimpose center dot and confidence center heatmap
    """
    out = superimpose_heatmap(img, center, alpha)
    if add_dot:
        out += superimpose_center_dot(img, center)
    return out

def superimpose_regression(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor) -> torch.Tensor:
    assert_img(img)
    assert_chw(center, "center")
    assert_chw(keypoints_regression, "keypoints_regression")
    assert_even_channels(keypoints_regression, "keypoints_regression")
    assert center.shape[0] == 1, f"Center tensor must have 1 channel, not {center.shape[0]}"
    assert center.shape[-2:] == keypoints_regression.shape[-2:], "Center and keypoints tensors (H, W) shape must be the same"

    cx, cy = get_center_pixel_xy(center)
    
    scale_factor_y = img.shape[-2] / center.shape[-2]
    scale_factor_x = img.shape[-1] / center.shape[-1]
    scaled_center_y = cy * scale_factor_y
    scaled_center_x = cx * scale_factor_x

    out = img.clone()
    for kpt_x, kpt_y in get_absolute_keypoints_xy(center, keypoints_regression):
        scaled_kpt_y = kpt_y * scale_factor_y
        scaled_kpt_x = kpt_x * scale_factor_x
        out = draw_vector(out, (scaled_center_x, scaled_center_y), (scaled_kpt_x, scaled_kpt_y), thickness=2)
    return out

def superimpose_hm_reg(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    out = superimpose_heatmap(img, heatmap, alpha)
    out = superimpose_regression(out, center, keypoints_regression)
    return out

def superimpose_hm_reg_score(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    assert_img(img)
    scores = get_hm_reg_alignment_scores(center, keypoints_regression, heatmap)
    scores *= 1.8  # Scaling up the values, after the scale up, scores are still <= 1
    return superimpose_heatmap(img, scores)

def superimpose_offset(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor, heatmap: torch.Tensor, offset: torch.Tensor, viz_scale: float = 1) -> torch.Tensor:
    assert_img(img)
    assert_chw(center, "center")
    assert_chw(keypoints_regression, "keypoints_regression")
    assert_chw(offset, "offset")
    assert_even_channels(keypoints_regression, "keypoints_regression")
    assert center.shape[0] == 1, f"Center tensor must have 1 channel, not {center.shape[0]}"
    assert keypoints_regression.shape[0] == offset.shape[0], f"Keypoint regression tensor must have the same number of channels as the offset tensor, instead of {keypoints_regression.shape[0]} and {offset.shape[0]}"
    assert center.shape[-2:] == keypoints_regression.shape[-2:] == offset.shape[-2:], "Center, keypoints, and offset tensors (H, W) shape must be the same"

    scale_factor_y = img.shape[-2] / center.shape[-2]
    scale_factor_x = img.shape[-1] / center.shape[-1]
    scores = get_hm_reg_alignment_scores(center, keypoints_regression, heatmap)

    out = img.clone()
    for (jx, jy), (ox, oy) in get_offsets_at_joint_locations_xy(scores, offset):
        scaled_kpt_y = jy * scale_factor_y
        scaled_kpt_x = jx * scale_factor_x
        scaled_offset_y = scaled_kpt_y + oy * scale_factor_y * viz_scale
        scaled_offset_x = scaled_kpt_x + ox * scale_factor_x * viz_scale
        out = draw_vector(out, (scaled_kpt_x, scaled_kpt_y), (scaled_offset_x, scaled_offset_y), thickness=2)
    return out


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    net = MoveNet(
        num_classes=13,
        width_mult=1.0,
        mode='train'
    ).cuda()

    state_dict = torch.load('checkpoints/mpii_pretrained_h36m_finetuned.pth')
    net.load_state_dict(state_dict)

    net.eval()
    
    data = Data(default_cfg)
    dataloader = data.getTestDataloader()

    with torch.no_grad():
        batch0, batch0_path = next(iter(dataloader))
        
        batch0 = batch0.cuda()

        print(f"Processing images {batch0_path}...")
        print("Input size: ", batch0.size())

        # out = (keypoint heatmaps, person centers, keypoint regressions, offsets)
        out = net(batch0)
        heatmaps, centers, regressions, offsets = out
        heatmaps_size, centers_size, regressions_size, offsets_size = heatmaps.size(), centers.size(), regressions.size(), offsets.size()

        B, k, h, w = heatmaps_size
        print(f"{B=}, {h=}, {w=}, {k=}")

        heatmap0, center0, regression0, offset0 = heatmaps[0].cpu(), centers[0].cpu(), regressions[0].cpu(), offsets[0].cpu()

        # Read the full-size original image
        img0 = torchvision.io.read_image(batch0_path[0]) / 255  # [0, 255] -> [0, 1]
        save_image(img0, args.output_dir / 'input_img.jpg')
        save_image(superimpose_heatmap(img0, heatmap0), args.output_dir / 'heatmap.jpg')
        save_image(superimpose_center(img0, center0), args.output_dir / 'center.jpg')
        save_image(superimpose_regression(img0, center0, regression0), args.output_dir / 'keypt_regression.jpg')
        save_image(superimpose_hm_reg(img0, center0, regression0, heatmap0), args.output_dir / 'hm_reg.jpg')
        save_image(superimpose_hm_reg_score(img0, center0, regression0, heatmap0), args.output_dir / 'hm_reg_score.jpg')
        save_image(superimpose_offset(img0, center0, regression0, heatmap0, offset0, viz_scale=1), args.output_dir / 'keypt_offsets.jpg')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("assets/MoveENet/imgs/viz_heads"))
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()  # relative to the invocation directory (cwd)
    return args

if __name__ == '__main__':
    main()