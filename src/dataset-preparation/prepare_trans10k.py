"""
Convert Trans10K binary masks to COCO instance segmentation JSON.

Trans10K masks are binary: 0 = background, 255 = transparent object.
Connected component analysis separates individual instances.

Usage:
    python src/dataset-preparation/prepare_trans10k.py --data_root /path/to/Trans10K
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage
from pycocotools import mask as mask_util


MIN_AREA = 100


def binary_mask_to_instances(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8)

    labeled, num_instances = ndimage.label(binary)
    instances = []
    for inst_id in range(1, num_instances + 1):
        inst_mask = (labeled == inst_id).astype(np.uint8)
        area = int(inst_mask.sum())
        if area < MIN_AREA:
            continue
        instances.append(inst_mask)
    return instances


def encode_rle(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def get_bbox(binary_mask):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


def convert_split(data_root, split, output_path):
    image_dir = os.path.join(data_root, split, "images")
    mask_dir = os.path.join(data_root, split, "masks")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    print(f"[{split}] Found {len(image_files)} images")

    images = []
    annotations = []
    ann_id = 0

    for img_id, fname in enumerate(image_files):
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, stem + ".png")

        if not os.path.exists(mask_path):
            print(f"  WARNING: mask not found for {fname}, skipping")
            continue

        img = Image.open(os.path.join(image_dir, fname))
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": f"{split}/images/{fname}",
            "height": h,
            "width": w,
        })

        instances = binary_mask_to_instances(mask_path)
        for inst_mask in instances:
            rle = encode_rle(inst_mask)
            bbox = get_bbox(inst_mask)
            area = int(inst_mask.sum())

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

        if (img_id + 1) % 500 == 0:
            print(f"  Processed {img_id + 1}/{len(image_files)} ...")

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "transparent_object"}],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_dict, f)

    print(f"[{split}] Saved {len(images)} images, {len(annotations)} annotations")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to Trans10K dataset root")
    parser.add_argument("--output_dir", default="datasets/trans10k", help="Output directory for COCO JSONs")
    args = parser.parse_args()

    for split, out_name in [("train", "coco_trans10k_train.json"), ("validation", "coco_trans10k_val.json")]:
        out_path = os.path.join(args.output_dir, out_name)
        convert_split(args.data_root, split, out_path)

    print("\nDone!")
