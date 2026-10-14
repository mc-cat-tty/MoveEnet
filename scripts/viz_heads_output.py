"""
@mc-cat-tty

Visualization script to create nice images of the four heads output.
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

def superimpose_heatmap(img: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    hm_one_channel = heatmap.amax(0, keepdim=True)
    hm_resized = F.interpolate(hm_one_channel.unsqueeze(0), size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=False)[0, 0]
    hm_u8 = (hm_resized*255).byte().numpy()
    hm_colormapped = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)  # BGR uint8 (H, W, 3)
    hm_colormapped = cv2.cvtColor(hm_colormapped, cv2.COLOR_BGR2RGB)
    hm_colormapped = torch.from_numpy(hm_colormapped).permute(2, 0, 1).float() / 255
    blending_mask = (hm_resized * alpha).unsqueeze(0)
    return (1-blending_mask) * img + blending_mask * hm_colormapped

def superimpose_center_dot(img: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    center_y = center.max(dim=-1).values.argmax()
    center_x = center.max(dim=-2).values.argmax()
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


def draw_vector(img: torch.Tensor, start_xy, end_xy, color=(255, 255, 255), thickness=1) -> torch.Tensor:
    canvas = np.ascontiguousarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))  # CHW [0,1] -> HWC uint8
    pt1 = (int(start_xy[0]), int(start_xy[1]))
    pt2 = (int(end_xy[0]), int(end_xy[1]))
    cv2.arrowedLine(canvas, pt1, pt2, color, thickness, tipLength=0.1)
    return torch.from_numpy(canvas).permute(2, 0, 1).float() / 255


def superimpose_regression(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor) -> torch.Tensor:
    center_y = center.max(dim=-1).values.argmax()
    center_x = center.max(dim=-2).values.argmax()
    keypoints = keypoints_regression[:, center_y, center_x]
    
    scale_factor_y = img.shape[-2] / center.shape[-2]
    scale_factor_x = img.shape[-1] / center.shape[-1]
    scaled_center_y = int(center_y * scale_factor_y)
    scaled_center_x = int(center_x * scale_factor_x)

    out = img.clone()
    for kpt_x, kpt_y in zip(keypoints[::2], keypoints[1::2]):
        scaled_kpt_y = int((kpt_y+center_y) * scale_factor_y)
        scaled_kpt_x = int((kpt_x+center_x) * scale_factor_x)
        out = draw_vector(out, (scaled_center_x, scaled_center_y), (scaled_kpt_x, scaled_kpt_y), thickness=2)
    return out

def superimpose_hm_reg(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    out = superimpose_heatmap(img, heatmap, alpha)
    out = superimpose_regression(out, center, keypoints_regression)
    return out

def superimpose_hm_reg_score(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    pass

def superimpose_offset(img: torch.Tensor, center: torch.Tensor, keypoints_regression: torch.Tensor, offset: torch.Tensor, viz_scale: float = 1) -> torch.Tensor:
    center_y = center.max(dim=-1).values.argmax()
    center_x = center.max(dim=-2).values.argmax()
    keypoints = keypoints_regression[:, center_y, center_x]

    assert center.shape[-2:] == offset.shape[-2:], "center and offset shape must be the same"
    
    scale_factor_y = img.shape[-2] / center.shape[-2]
    scale_factor_x = img.shape[-1] / center.shape[-1]

    out = img.clone()
    for idx, (kpt_x, kpt_y) in enumerate(zip(keypoints[::2], keypoints[1::2])):
        kpt_x, kpt_y = center_x + int(kpt_x+0.5), center_y + int(kpt_y+0.5)  # computing absolute kpt position
        kpt_x, kpt_y = min(max(0, kpt_x), offset.shape[-1]-1), min(max(0, kpt_y), offset.shape[-2]-1)  # clamping
        offset_x, offset_y = offset[2*idx, kpt_y, kpt_x], offset[2*idx+1, kpt_y, kpt_x]
        scaled_kpt_y = int(kpt_y * scale_factor_y)
        scaled_kpt_x = int(kpt_x * scale_factor_x)
        scaled_offset_y = scaled_kpt_y + offset_y * scale_factor_y * viz_scale
        scaled_offset_x = scaled_kpt_x + offset_x * scale_factor_x * viz_scale
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

    state_dict = torch.load('../hpe-core/example/movenet/models/e97_valacc0.81209.pth')
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

        # batch0 = batch0.cpu()
        # img0 = batch0[0] / 255  # [0, 255] -> [0, 1]
        img0 = torchvision.io.read_image(batch0_path[0]) / 255
        save_image(img0, args.output_dir / 'input_img.jpg')
        save_image(superimpose_heatmap(img0, heatmap0), args.output_dir / 'heatmap.jpg')
        save_image(superimpose_center(img0, center0), args.output_dir / 'center.jpg')
        save_image(superimpose_regression(img0, center0, regression0), args.output_dir / 'keypt_regression.jpg')
        save_image(superimpose_hm_reg(img0, center0, regression0, heatmap0), args.output_dir / 'hm_reg.jpg')
        # save_image(superimpose_hm_reg_score(img0, center0, regression0, heatmap0), args.output_dir / 'hm_reg_score.jpg')
        save_image(superimpose_offset(img0, center0, regression0, offset0, viz_scale=1), args.output_dir / 'keypt_offsets.jpg')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("viz_heads"))
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()  # relative to the invocation directory (cwd)
    return args

if __name__ == '__main__':
    main()