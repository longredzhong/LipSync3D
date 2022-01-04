"""
Following https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/combine_A_and_B.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(".")

from models import networks
from options.options import Options
from audiodvp_utils.util import create_dir, load_coef, get_file_list, load_face_emb, get_max_crop_region
from audiodvp_utils.rescale_image import rescale_and_paste


if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.data_dir, 'mask'))

    alpha_list = load_coef(os.path.join(opt.data_dir, 'alpha'))
    beta_list = load_coef(os.path.join(opt.data_dir, 'beta'))
    delta_list = load_coef(os.path.join(opt.data_dir, 'delta'))
    gamma_list = load_coef(os.path.join(opt.data_dir, 'gamma'))
    angle_list = load_coef(os.path.join(opt.data_dir, 'rotation'))
    translation_list = load_coef(os.path.join(opt.data_dir, 'translation'))
    face_emb_list = load_face_emb(opt.data_dir)

    crop_region_list = load_coef(os.path.join(opt.data_dir, 'crop_region'))
    full_image_list = get_file_list(os.path.join(opt.data_dir, 'full'))
    
    top, bottom, left, right = get_max_crop_region(crop_region_list)
    H, W, _ = cv2.imread(full_image_list[0]).shape
    mouth_mask = networks.MouthMask(opt).to(opt.device)

    for i in tqdm(range(len(alpha_list))):
        alpha = alpha_list[i].unsqueeze(0).cuda()
        beta = beta_list[i].unsqueeze(0).cuda()
        delta = delta_list[i].unsqueeze(0).cuda()
        gamma = gamma_list[i].unsqueeze(0).cuda()
        rotation = angle_list[i].unsqueeze(0).cuda()
        translation = translation_list[i].unsqueeze(0).cuda()
        face_emb = face_emb_list[i].unsqueeze(0).cuda()
        crop_region = crop_region_list[i]
        empty_image = np.zeros((H, W), np.uint8)

        mask = mouth_mask(alpha, delta, beta, gamma, rotation, translation, face_emb)
        mask = mask.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255.0
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4)
        rescaled_mask = rescale_and_paste(crop_region, empty_image, mask)
        rescaled_mask = rescaled_mask[top:bottom, left:right]
        rescaled_mask = cv2.resize(rescaled_mask, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(os.path.join(opt.data_dir, 'mask', '%05d.png' % (i+1)), rescaled_mask)

    create_dir(os.path.join(opt.data_dir, 'nfr', 'A', 'train'))
    create_dir(os.path.join(opt.data_dir, 'nfr', 'B', 'train'))

    masks = get_file_list(os.path.join(opt.data_dir, 'mask'))
    renders = get_file_list(os.path.join(opt.data_dir, 'render'))

    for i in tqdm(range(len(masks))):
        mask = cv2.imread(masks[i])
        render = cv2.imread(renders[i])
        full = cv2.imread(full_image_list[i])
        crop_region = crop_region_list[i]

        empty_image = np.zeros((H, W, 3), np.uint8)

        rescaled_render = rescale_and_paste(crop_region, empty_image, render)
        rescaled_render = rescaled_render[top:bottom, left:right]
        rescaled_render = cv2.resize(rescaled_render, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        
        rescaled_crop = full[top:bottom, left:right]
        rescaled_crop = cv2.resize(rescaled_crop, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        
        masked_crop = cv2.bitwise_and(rescaled_crop, mask)
        masked_render = cv2.bitwise_and(rescaled_render, mask)

        cv2.imwrite(os.path.join(opt.data_dir, 'nfr', 'A', 'train', '%05d.png' % (i+1)), masked_crop)
        cv2.imwrite(os.path.join(opt.data_dir, 'nfr', 'B', 'train', '%05d.png' % (i+1)), masked_render)

    splits = os.listdir(os.path.join(opt.data_dir, 'nfr', 'A'))

    for sp in splits:
        image_fold_A = os.path.join(os.path.join(opt.data_dir, 'nfr', 'A'), sp)
        image_fold_B = os.path.join(os.path.join(opt.data_dir, 'nfr', 'B'), sp)
        image_list = os.listdir(image_fold_A)

        image_fold_AB = os.path.join(opt.data_dir, 'nfr', 'AB', sp)
        if not os.path.isdir(image_fold_AB):
            os.makedirs(image_fold_AB)

        for n in tqdm(range(len(image_list))):
            name_A = image_list[n]
            path_A = os.path.join(image_fold_A, name_A)

            name_B = name_A
            path_B = os.path.join(image_fold_B, name_B)

            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_AB = name_A
                path_AB = os.path.join(image_fold_AB, name_AB)
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)
