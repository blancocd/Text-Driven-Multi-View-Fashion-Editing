import os
import sys
import gc
import json

from utils.concat import concatenate_imgs, transp_to_white
from utils.create_masks_from_seg import get_mask_4ddress
from utils.deconcat import deconcat_img
from segmentation.segment_dir import segment_dir
import logging
from PIL import Image
from huggingface_hub import login
import numpy as np
from diffusers import FluxKontextPipeline, FluxFillPipeline
import torch

# token = os.getenv("HUGGINGFACE_TOKEN")
# login(token=token)

import random
MAX_SEED = np.iinfo(np.int32).max

def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

def remove_garment_kontext(pipe, image, prompt, negative_prompt=None, true_cfg_scale=1.0, num_inference_steps=28, guidance_scale=3.5, seed = None):
    h, w = (image.height, image.width) if isinstance(image, Image.Image) else (image.shape[0], image.shape[1])
    seed =  seed or random.randint(0, MAX_SEED)
    print(f'Flux Kontext seed is {seed}')
    gen_image = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        height=h,
        width=w,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]
    
    torch.cuda.empty_cache()
    return gen_image

def remove_garment_fill(pipe, image, mask, prompt, seed = None):
    h, w = (image.height, image.width) if isinstance(image, Image.Image) else (image.shape[0], image.shape[1])
    seed =  seed or random.randint(0, MAX_SEED)
    print(f'Flux Fill seed is {seed}')
    gen_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]
    
    torch.cuda.empty_cache()
    return gen_image

def get_equally_spaced_anchors_indices(initial_anchor_idx, num_views, num_anchors):
    anchor_indices = [(initial_anchor_idx + (i * round(num_views / num_anchors))) % num_views for i in range(num_anchors)]
    
    anchor_groups = {i: [] for i in anchor_indices}
    for index_to_assign in range(num_views):
        min_distance = float('inf')
        closest_anchor = -1
        for anchor in anchor_indices:
            direct_dist = abs(index_to_assign - anchor)
            wrap_dist = num_views - direct_dist
            distance = min(direct_dist, wrap_dist)
            if distance < min_distance:
                min_distance = distance
                closest_anchor = anchor
        anchor_groups[closest_anchor].append(index_to_assign)

    indices_list, indices_to_gen_save_flag_list = [anchor_indices], [[False] + ([True] * (len(anchor_indices)-1))]
    for anchor_idx in anchor_indices:
        indices = anchor_groups[anchor_idx]
        indices_to_gen_save = [i!=anchor_idx for i in indices]
        if any(indices_to_gen_save):
            indices_list.append(indices)
            indices_to_gen_save_flag_list.append(indices_to_gen_save)

    return indices_list, indices_to_gen_save_flag_list
    

def get_sweeping_anchors_indices(initial_anchor_idx, num_views):
    curr_anchors = [initial_anchor_idx, initial_anchor_idx]
    completed_indices = [initial_anchor_idx]
    indices_list, indices_to_gen_save_flag_list = [], []
    while len(completed_indices) != num_views:
        indices = list(range(curr_anchors[0]-2, curr_anchors[0]+1)) + list(range(curr_anchors[1], curr_anchors[1]+3))
        indices = list(dict.fromkeys([(i + num_views) % num_views for i in indices]))
        curr_anchors = [indices[0], indices[-1]]
        indices_to_gen_save = [i not in completed_indices for i in indices]
        completed_indices.extend([i for i in indices if i not in completed_indices])
        indices_list.append(indices)
        indices_to_gen_save_flag_list.append(indices_to_gen_save)
        
    sorted_pairs = sorted(zip(indices_list[-1], indices_to_gen_save_flag_list[-1]), key=lambda pair: pair[0], reverse=True)
    indices_list[-1], indices_to_gen_save_flag_list[-1] = zip(*sorted_pairs)
    return indices_list, indices_to_gen_save_flag_list

def get_mvadapter_indices():
    indices = [4, 5, 0, 1, 2, 3]
    to_save = [True, True, False, True, True, True]
    return [indices], [to_save]


def remove_garment_anchors(scan_dir, scan_out_dir, garment_type, initial_anchor_idx, indices_list, 
                           indices_to_gen_save_flag_list, flux_kontext_args, flux_fill_args, 
                           ratio=4, pixel_sep=20, dil_its=1, ero_its=1, verbose = False):
    if scan_dir == scan_out_dir:
        print("Images will be overwritten! exiting.")
        return
    
    os.makedirs(os.path.join(scan_out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(scan_out_dir, "segmentation_masks"), exist_ok=True)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    def vcomment(msg):
        if verbose:
            logger.info(msg)

    # Preparing sets of images that will be concatenated together
    img_dir = os.path.join(scan_dir, 'images')
    img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])

    # Load FluxKontext
    pipe_kontext = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16, safety_checker=None).to("cuda")
    pipe_kontext.safety_checker = disabled_safety_checker

    # Remove garment from view with Flux Kontext. White bg works better with these models.
    front_view_img = transp_to_white(Image.open(os.path.join(scan_dir, 'images', img_fns[initial_anchor_idx])))
    gen_front_view_img = remove_garment_kontext(pipe_kontext, front_view_img, **flux_kontext_args)
    gen_front_view_img.save(os.path.join(scan_out_dir, 'images', f'train_{initial_anchor_idx:04d}.png'))
    vcomment(f'Removed garment from front view image: {initial_anchor_idx} and saved it.')
    del pipe_kontext; del gen_front_view_img; gc.collect(); torch.cuda.empty_cache()

    # Load FluxFill pipeline
    pipe_fill = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, safety_checker=None).to("cuda")
    pipe_fill.safety_checker = disabled_safety_checker

    vcomment(f"Starting {len(indices_list)} iterations to generate all views:")
    for indices, indices_to_gen_save_flag in zip(indices_list, indices_to_gen_save_flag_list):
        anchor_indices = [i for i, f in zip(indices, indices_to_gen_save_flag) if not f]
        indices_to_gen_save = [i for i, f in zip(indices, indices_to_gen_save_flag) if f]
        vcomment(f"Anchor indices {anchor_indices} will be used to generate {indices_to_gen_save}.")
        
        # Loading images based on whether they are without garment and thus anchor or to be generated
        concat_imgs, concat_segs = [], []
        for i, gen_save, in zip(indices, indices_to_gen_save_flag):
            # Choose images to be concatenated, could be from scan_dir or from the current out_dir
            if gen_save:
                img_fn = os.path.join(scan_dir, 'images', f'train_{i:04d}.png')
                concat_imgs.append(Image.open(img_fn))
                seg_fn = os.path.join(scan_dir, 'segmentation_masks', f'train_{i:04d}.png')
                concat_segs.append(Image.open(seg_fn))
            else:
                img_fn = os.path.join(scan_out_dir, 'images', f'train_{i:04d}.png')
                concat_imgs.append(Image.open(img_fn))
                concat_segs.append(None)

        # Concatenate images and segmentation maps
        concat_img, concat_seg, concat_img_coords_list, human_dims_list = concatenate_imgs(
            concat_imgs, concat_segs, ratio=ratio, pixel_sep=pixel_sep)
        vcomment("Concatenated views.")

        # Get mask of concatenated anchor images according to type(s) of inpainting
        mask = get_mask_4ddress(concat_seg, garment_type, dil_its=dil_its, ero_its=ero_its).astype(np.float32)

        vcomment(f"Mask of type(s): {garment_type} is ready for concatenated views.")

        # Inpaint concatentated anchor images with FluxFill:
        concat_img_pil = Image.fromarray(concat_img.astype(np.uint8))
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        gen_concat_images = remove_garment_fill(pipe_fill, concat_img_pil, mask_pil, **flux_fill_args)
        vcomment(f"Removed garments from concatenated views.")
        
        # Deconcatenate images and save generated images
        deconcat_img(scan_out_dir, gen_concat_images, indices, concat_img_coords_list, human_dims_list, 
                                  indices_to_gen_save_flag=indices_to_gen_save_flag)
        vcomment(f"{indices_to_gen_save} have been saved to scan directory.")
        del gen_concat_images; gc.collect(); torch.cuda.empty_cache()
    segment_dir(scan_out_dir)
    del pipe_fill; gc.collect(); torch.cuda.empty_cache()


def get_initial_anchor_idx(scan_dir, img_fns):
    human_counts = []
    hair_counts = []
    for img_fn in img_fns:
        seg_path = os.path.join(scan_dir, 'segmentation_masks', img_fn)
        seg_map = np.array(Image.open(seg_path).convert('RGB'))

        # Assuming your get_mask_4ddress function is available
        human_mask = get_mask_4ddress(seg_map, 'human') 
        hair_mask = get_mask_4ddress(seg_map, 'hair')

        human_counts.append(human_mask.sum())
        hair_counts.append(hair_mask.sum())
        
    # Convert lists to NumPy arrays for efficient operations
    human_counts = np.array(human_counts)
    hair_counts = np.array(hair_counts)

    sorted_indices_by_human = np.argsort(human_counts)
    top_2_indices = sorted_indices_by_human[-2:]
    
    hair_counts_of_top_2 = hair_counts[top_2_indices]
    winner_index_in_top_2 = np.argmin(hair_counts_of_top_2)
    
    # 5. Use that index to pick the final winner from our top_2_indices array
    final_anchor_idx = top_2_indices[winner_index_in_top_2]
    
    return final_anchor_idx

def remove_garments(dataset_dir, out_dir, garment_data_json, index, 
                    outer_dil_its=1, outer_ero_its=1,
                    inner_dil_its=1, inner_ero_its=1):
    with open(garment_data_json, 'r') as f:
        garment_data = json.load(f)
    scan_names = list(garment_data.keys())
    scan_name = scan_names[index-1]
    print(scan_name)
    scan_dict = garment_data[scan_name]
    scan_dir = os.path.join(dataset_dir, scan_name)

    initial_anchor_idx = scan_dict['anchor_idx']
    img_dir = os.path.join(scan_dir, 'images')
    img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])
    # indices_list, indices_to_gen_save_flag_list = get_sweeping_anchors_indices(initial_anchor_idx, len(img_fns))
    indices_list, indices_to_gen_save_flag_list = get_equally_spaced_anchors_indices(initial_anchor_idx, len(img_fns), 4)
    # indices_list, indices_to_gen_save_flag_list = get_mvadapter_indices()
    
    # Removing outer garment
    garment_type = 'outer'
    if garment_type in scan_dict['flux_kontext_args']:
        scan_noouter_dir = os.path.join(out_dir, scan_name, garment_type)
        flux_kontext_args = scan_dict['flux_kontext_args'][garment_type]
        flux_fill_args = scan_dict['flux_fill_args'][garment_type]

        remove_garment_anchors(scan_dir, scan_noouter_dir, 'outer', initial_anchor_idx, indices_list, 
                            indices_to_gen_save_flag_list, flux_kontext_args, flux_fill_args, 
                            verbose=True, dil_its=outer_dil_its, ero_its=outer_ero_its)
    else:
        scan_noouter_dir = scan_dir

    # Removing inner garment
    garment_type = 'inner'
    scan_noinner_dir = os.path.join(out_dir, scan_name, garment_type)
    flux_kontext_args = scan_dict['flux_kontext_args'][garment_type]
    flux_fill_args = scan_dict['flux_fill_args'][garment_type]

    remove_garment_anchors(scan_noouter_dir, scan_noinner_dir, garment_type, initial_anchor_idx, indices_list, 
                        indices_to_gen_save_flag_list, flux_kontext_args, flux_fill_args, 
                        verbose=True, dil_its=inner_dil_its, ero_its=inner_ero_its)
    
            
if __name__ == "__main__":
    if len(sys.argv) != 9:
        print(f"Usage: python {sys.argv[0]} <dataset_dir> <out_dir> <garment_data_json> <index> <outer_dil_its> <outer_ero_its> <inner_dil_its> <inner_ero_its>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    out_dir = sys.argv[2]
    garment_data_json = sys.argv[3]
    index = int(sys.argv[4])
    outer_dil_its = int(sys.argv[5])
    outer_ero_its = int(sys.argv[6])
    inner_dil_its = int(sys.argv[7])
    inner_ero_its = int(sys.argv[8])

    remove_garments(dataset_dir, out_dir, garment_data_json, index, 
                    outer_dil_its=outer_dil_its, outer_ero_its=outer_ero_its,
                    inner_dil_its=inner_dil_its, inner_ero_its=inner_ero_its)
