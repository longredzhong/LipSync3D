"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""

import os
import cv2
import argparse
from torch import full
from tqdm import tqdm
import numpy as np
from .util import load_coef, create_dir, get_file_list

# Resacle overlay and render images. Paste it on full image.
def rescale_and_paste(crop_region, full_image, target_image):
    top, bottom, left, right = crop_region
    height = bottom - top
    width = right - left
    
    pasted_image = full_image.copy()
    rescaled_target = cv2.resize(target_image, (width, height), interpolation=cv2.INTER_AREA)
    pasted_image[top:bottom, left:right] = rescaled_target

    return pasted_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    crop_region_list = load_coef(os.path.join(args.data_dir, 'crop_region'))
    full_image_list = get_file_list(os.path.join(args.data_dir, 'full'))
    overlay_image_list = get_file_list(os.path.join(args.data_dir, 'overlay'))
    render_image_list = get_file_list(os.path.join(args.data_dir, 'render'))

    create_dir(os.path.join(args.data_dir, 'rescaled_overlay'))
    create_dir(os.path.join(args.data_dir, 'rescaled_render'))

    for i in tqdm(range(len(full_image_list))):
        full_image = cv2.imread(full_image_list[i])
        overlay_image = cv2.imread(overlay_image_list[i])
        render_image = cv2.imread(render_image_list[i])
        crop_region = crop_region_list[i]

        H, W, _ = full_image.shape
        empty_image = np.zeros((H, W, 3), np.uint8)

        pasted_overlay = rescale_and_paste(crop_region, full_image, overlay_image)
        pasted_render = rescale_and_paste(crop_region, empty_image, render_image)

        cv2.imwrite(os.path.join(args.data_dir, 'rescaled_overlay', os.path.basename(full_image_list[i])), pasted_overlay)
        cv2.imwrite(os.path.join(args.data_dir, 'rescaled_render', os.path.basename(full_image_list[i])), pasted_render)   