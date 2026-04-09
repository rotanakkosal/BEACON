import os
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from glob import glob
from tqdm import tqdm

# --- CONFIG ---
DATA_ROOT = "datasets/clearpose_dataset"
TRAIN_SETS = ["set1", "set2", "set3", "set4", "set5", "set6"]
VAL_SETS = ["set7", "set8", "set9"]

def get_instance_masks(mask_path):
    """
    Reads a ClearPose label image and extracts individual instance masks.

    ClearPose format: Each pixel value represents an object ID.
    - 0 = background
    - 1, 2, 3, ... = different object instances

    Returns:
        List of (rle, area, bbox) tuples for each instance
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []

    # Find unique object IDs (exclude background = 0)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]

    instances = []
    for obj_id in obj_ids:
        # Create binary mask for this specific instance
        binary_mask = (mask == obj_id).astype(np.uint8)

        # Encode to COCO RLE format
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')

        # Calculate area and bbox
        area = int(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()

        # Skip very small annotations (noise)
        if area < 50:
            continue

        instances.append((rle, area, bbox))

    return instances

def convert_dataset():
    # Initialize COCO dictionaries
    coco_train = {
        "images": [], "annotations": [], 
        "categories": [{"id": 1, "name": "transparent_object"}]
    }
    coco_val = {
        "images": [], "annotations": [], 
        "categories": [{"id": 1, "name": "transparent_object"}]
    }

    global_img_id = 0
    global_ann_id = 0

    # Get all set folders
    set_folders = sorted(glob(os.path.join(DATA_ROOT, "set*")))

    print(f"Found sets: {[os.path.basename(s) for s in set_folders]}")

    for set_folder in set_folders:
        set_name = os.path.basename(set_folder)
        
        # Decide if TRAIN or VAL
        if set_name in TRAIN_SETS:
            target_dict = coco_train
            mode = "TRAIN"
        elif set_name in VAL_SETS:
            target_dict = coco_val
            mode = "VAL"
        else:
            print(f"Skipping unknown set: {set_name}")
            continue

        # Each set has scene folders (e.g., scene1, scene2)
        scene_folders = sorted(glob(os.path.join(set_folder, "*")))
        
        for scene_folder in tqdm(scene_folders, desc=f"Processing {set_name} ({mode})"):
            # Get RGB images
            rgb_files = sorted(glob(os.path.join(scene_folder, "*-color.png")))
            
            for img_path in rgb_files:
                # 1. Add Image Info
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                
                # Create unique file name relative to dataset root
                # e.g., set1/scene1/000000-color.png
                file_name = os.path.relpath(img_path, DATA_ROOT)
                
                image_info = {
                    "id": global_img_id,
                    "file_name": file_name,
                    "height": h,
                    "width": w
                }
                target_dict["images"].append(image_info)

                # 2. Find matching mask file
                # ClearPose format: 000000-color.png -> 000000-label.png
                prefix = os.path.basename(img_path).replace("-color.png", "")
                mask_path = os.path.join(scene_folder, f"{prefix}-label.png")

                if not os.path.exists(mask_path):
                    continue

                # 3. Extract all instance masks from the label image
                instances = get_instance_masks(mask_path)

                for rle, area, bbox in instances:
                    ann = {
                        "id": global_ann_id,
                        "image_id": global_img_id,
                        "category_id": 1,  # Unified "Transparent" class
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    target_dict["annotations"].append(ann)
                    global_ann_id += 1
                
                global_img_id += 1

    # Save JSONs
    print("Saving Train JSON...")
    with open(os.path.join(DATA_ROOT, "coco_clearpose_train.json"), "w") as f:
        json.dump(coco_train, f)
        
    print("Saving Val JSON...")
    with open(os.path.join(DATA_ROOT, "coco_clearpose_val.json"), "w") as f:
        json.dump(coco_val, f)

    print(f"Done! Train Images: {len(coco_train['images'])}, Val Images: {len(coco_val['images'])}")

if __name__ == "__main__":
    convert_dataset()