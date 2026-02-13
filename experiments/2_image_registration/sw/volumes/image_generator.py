#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import kornia.geometry.transform as KT
import math
from datetime import datetime
import imageio.v2 as imageio


def random_rigid_matrix_2d(tx_range, ty_range, angle_range_deg, device="cpu"):
    """
    2D rigid transform: rotation + translation (no scaling, no shear)
    """
    tx = np.random.uniform(*tx_range)
    ty = np.random.uniform(*ty_range)
    angle_deg = np.random.uniform(*angle_range_deg)

    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)

    A = torch.tensor([
        [c, -s, tx],
        [s,  c, ty],
    ], dtype=torch.float32, device=device)

    return A.unsqueeze(0), tx, ty, angle_deg


def apply_rigid_transform_2d_uint8(img_np, rigid_matrix_2x3):
    """
    img_np: (H, W) uint8
    rigid_matrix_2x3: (1, 2, 3)
    """
    img = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    _, _, H, W = img.shape

    warped = KT.warp_affine(
        img,
        rigid_matrix_2x3,
        dsize=(H, W),
        align_corners=True,
    )

    warped = warped[0, 0].cpu().numpy()
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    return warped


def save_duplicated_images(img, out_dir, n=256):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        imageio.imwrite(os.path.join(out_dir, f"IM{i}.png"), img)


def main():
    parser = argparse.ArgumentParser(
        description="Create reference/floating folders from IM1.png (2D rigid transform)."
    )
    parser.add_argument("--input_image", default="IM1.png", help="Input image (default: IM1.png)")
    parser.add_argument("--output_path", default=".", help="Base output folder (default: current)")
    parser.add_argument("--num_images", type=int, default=256, help="Number of copies (default: 256)")
    args = parser.parse_args()

    # Load image
    if not os.path.exists(args.input_image):
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    img = imageio.imread(args.input_image)
    if img.ndim != 2:
        raise ValueError("Input image must be grayscale (single channel)")

    # Generate random rigid transform
    rigid_matrix, tx, ty, angle = random_rigid_matrix_2d(
        tx_range=(-10, 10),
        ty_range=(-10, 10),
        angle_range_deg=(-10, 10),
    )

    # Apply transform
    img_deformed = apply_rigid_transform_2d_uint8(img, rigid_matrix)

    # Output dirs
    ref_dir = os.path.join(args.output_path, "reference")
    flo_dir = os.path.join(args.output_path, "floating")

    # Write IM0..IM255
    save_duplicated_images(img, ref_dir, n=args.num_images)
    save_duplicated_images(img_deformed, flo_dir, n=args.num_images)

    # Log
    log_path = os.path.join(args.output_path, "deformation_report.log")
    with open(log_path, "a") as f:
        f.write(
            f"[{datetime.now().isoformat()}] "
            f"{args.input_image}: "
            f"tx={tx:.2f}, ty={ty:.2f}, angle={angle:.2f} deg, "
            f"written={args.num_images}\n"
        )

    print(f"✅ reference: IM0..IM{args.num_images-1} written from {args.input_image}")
    print(f"✅ floating:  IM0..IM{args.num_images-1} written from deformed image")
    print(f"📝 Log written to: {log_path}")


if __name__ == "__main__":
    main()
