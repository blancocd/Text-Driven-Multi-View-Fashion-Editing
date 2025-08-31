import json
import os
from skimage.metrics import structural_similarity
import numpy as np
from utils.create_masks_from_seg import get_mask_4ddress
from utils.concat import transp_to_white
from PIL import Image
import numpy as np
import sys
import re
from tqdm import tqdm

def get_masked_ssim(img_1_fn, img_2_fn, masks):
    _, ssim_map = structural_similarity(np.array(Image.open(img_1_fn).convert('L')), 
                                        np.array(Image.open(img_2_fn).convert('L')), full=True)
    to_return = []
    for mask in masks:
        if mask.sum()==0:
            to_return.append(-1)
        else:
            to_return.append(np.mean(ssim_map[mask]))
    return to_return

def get_masked_psnr(img_1_fn, img_2_fn, masks):
    img_1_arr = np.array(Image.open(img_1_fn).convert('RGB')).astype(np.float32) / 255.
    img_2_arr = np.array(Image.open(img_2_fn).convert('RGB')).astype(np.float32) / 255.
    squared_diff_map = (img_1_arr - img_2_arr) ** 2
    to_return = []
    for mask in masks:
        if mask.sum()==0:
            to_return.append(-1)
        else:
            mse_masked = np.mean(squared_diff_map[mask])
            psnr_masked = 10 * np.log10(1. / mse_masked)
            to_return.append(psnr_masked)
    return to_return

fourddress_palette = np.array([
    [128., 128., 128.],   # 0 skin
    [255., 128.,   0.],   # 1 hair
    [128.,   0., 255.],   # 2 shoes
    [  0., 255.,   0.],   # 3 lower
])

def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection, union

def get_ious(segformer_fn, gen_segformer_fn, dress):
    segformer_map = np.array(transp_to_white(Image.open(segformer_fn).convert('RGB')))
    gen_segformer_map = np.array(Image.open(gen_segformer_fn).convert('RGB'))
    union_count, intersection_count = 0, 0
    for target_color in fourddress_palette[1:-1]:
        segformer_mask = np.all(np.abs(segformer_map - target_color) <= 5, axis=-1)
        gen_segformer_mask = np.all(np.abs(gen_segformer_map - target_color) <= 5, axis=-1)
        intersection, union = iou(segformer_mask, gen_segformer_mask)
        intersection_count += intersection
        union_count += union
    
    if not dress:
        # inner
        target_color = [255.,   0.,   0.]
        segformer_mask = np.all(np.abs(segformer_map - target_color) <= 5, axis=-1)
        gen_segformer_mask = np.all(np.abs(gen_segformer_map - target_color) <= 5, axis=-1)

        rows = np.any(segformer_mask, axis=1)
        last_row_with_inner_garment = np.where(rows)[0][-1] + 1

        # lower
        target_color = fourddress_palette[-1]
        segformer_mask = np.all(np.abs(segformer_map - target_color) <= 5, axis=-1)
        gen_segformer_mask = np.all(np.abs(gen_segformer_map - target_color) <= 5, axis=-1)
        intersection, union = iou(segformer_mask[last_row_with_inner_garment:, :], gen_segformer_mask[last_row_with_inner_garment:, :])
        intersection_count += intersection
        union_count += union


        # skin
        target_color = fourddress_palette[0]
        segformer_mask = np.all(np.abs(segformer_map - target_color) <= 5, axis=-1)
        gen_segformer_mask = np.all(np.abs(gen_segformer_map - target_color) <= 5, axis=-1)
        intersection, union = iou(segformer_mask[last_row_with_inner_garment:, :], gen_segformer_mask[last_row_with_inner_garment:, :])
        intersection_count += intersection
        union_count += union

    return intersection_count/union_count

def main(gen_method_dir, dataset_dir, garment_data_json):
    with open(garment_data_json, 'r') as f:
        garment_data = json.load(f)
    scan_names = list(garment_data.keys())
    
    method = os.path.basename(os.path.normpath(gen_method_dir))
    match = re.search(r'(\d+)_(\d+)_(\d+)_(\d+)', method)
    outer_dil_its = int(match.group(1))
    outer_ero_its = int(match.group(2))

    results_dict = {}
    for scan_name in tqdm(scan_names):
        dress = "dress" in garment_data[scan_name]["flux_fill_args"]["inner"]["prompt"]
        img_fns = sorted([f for f in os.listdir(os.path.join(gen_method_dir, scan_name, 'inner', 'images')) if f.startswith('train')])
        # For outer garment case the inner garment shouldn't be changed, so there are inner_mask and nongen_mask
        # For inner garment case there is no inner garment to preserve, so there is only nongen_mask
        results_dict[scan_name] = {
            'indices': [],
            'ssim_inner': [],
            'ssim_nongen_remove_outer': [],
            'ssim_nongen_remove_inner': [],
            'psnr_inner': [],
            'psnr_nongen_remove_outer': [],
            'psnr_nongen_remove_inner': [],
            'ious_orig-remove_outer': [],
            'ious_orig-remove_inner': []
        }
        for img_fn in img_fns:
            idx = int(img_fn.split('_')[1].split('.')[0])
            results_dict[scan_name]['indices'].append(idx)

            img_path = os.path.join(dataset_dir, scan_name, 'images', img_fn)
            seg_path = os.path.join(dataset_dir, scan_name, 'segmentation_masks', img_fn)
            segformer_path = os.path.join(dataset_dir, scan_name, 'segformer_segmentation_masks', img_fn)
            seg_map = np.array(Image.open(seg_path).convert('RGB'))
            human_mask = get_mask_4ddress(seg_map, 'human', dil_its=0, ero_its=None)
            
            gen_scan_dir = os.path.join(gen_method_dir, scan_name)
            if os.path.isdir(os.path.join(gen_scan_dir, 'outer')):
                gen_remove_outer_img_path = os.path.join(gen_scan_dir, 'outer', 'images', img_fn)
                inner_mask = get_mask_4ddress(seg_map, 'inner', dil_its=0, ero_its=None)
                upper_mask = get_mask_4ddress(seg_map, 'upper', dil_its=outer_dil_its, ero_its=outer_ero_its)

                shouldnt_have_edited_mask_remove_outer = np.logical_and(human_mask, ~upper_mask)

                ssim_inner, ssim_nongen_remove_outer = get_masked_ssim(img_path, gen_remove_outer_img_path, [inner_mask, shouldnt_have_edited_mask_remove_outer])
                psnr_inner, psnr_nongen_remove_outer = get_masked_psnr(img_path, gen_remove_outer_img_path, [inner_mask, shouldnt_have_edited_mask_remove_outer])

                results_dict[scan_name]['ssim_inner'].append(f"{ssim_inner:.6f}")
                results_dict[scan_name]['ssim_nongen_remove_outer'].append(f"{ssim_nongen_remove_outer:.6f}")
                results_dict[scan_name]['psnr_inner'].append(f"{psnr_inner:.6f}")
                results_dict[scan_name]['psnr_nongen_remove_outer'].append(f"{psnr_nongen_remove_outer:.6f}")

                gen_remove_outer_segformer_path = os.path.join(gen_scan_dir, 'outer', 'segmentation_masks', img_fn)
                iou_outer = get_ious(segformer_path, gen_remove_outer_segformer_path, dress)
                results_dict[scan_name]['ious_orig-remove_outer'].append(iou_outer)

            gen_remove_inner_img_path = os.path.join(gen_scan_dir, 'inner', 'images', img_fn)
            cloth_mask = get_mask_4ddress(seg_map, ['outer', 'inner', 'lower'], dil_its=outer_dil_its, ero_its=outer_ero_its)
            shouldnt_have_edited_mask_remove_inner = np.logical_and(human_mask, ~cloth_mask)

            ssim_nongen_remove_inner = get_masked_ssim(img_path, gen_remove_inner_img_path, [shouldnt_have_edited_mask_remove_inner])
            psnr_nongen_remove_inner = get_masked_psnr(img_path, gen_remove_inner_img_path, [shouldnt_have_edited_mask_remove_inner])

            results_dict[scan_name]['ssim_nongen_remove_inner'].append(f"{ssim_nongen_remove_inner[0]:.6f}")
            results_dict[scan_name]['psnr_nongen_remove_inner'].append(f"{psnr_nongen_remove_inner[0]:.6f}")

            gen_remove_inner_segformer_fn = os.path.join(gen_scan_dir, 'inner', 'segmentation_masks', img_fn)
            iou_inner = get_ious(segformer_path, gen_remove_inner_segformer_fn, dress)
            results_dict[scan_name]['ious_orig-remove_inner'].append(iou_inner)
                

    results_fn = f'{method}_cpu_results.json'
    print(results_fn)
    with open(results_fn, 'w') as f:
        json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Example usage: {sys.argv[0]} <gen_method_dir> <dataset_dir> <garment_data>")
    else:
        gen_method_dir = sys.argv[1]
        dataset_dir = sys.argv[2]
        garment_data_json = sys.argv[3]
        print(sys.argv)
        main(gen_method_dir, dataset_dir, garment_data_json)
