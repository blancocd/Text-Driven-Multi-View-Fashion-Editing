from PIL import Image
from remove_garment_mv import remove_garment_kontext
from utils.concat import transp_to_white
import os
import gc
import sys
import random
import numpy as np
import json
from diffusers import FluxKontextPipeline
import torch
MAX_SEED = np.iinfo(np.int32).max

def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

# Test the seeds and prompts that will be used for multiview generation.
# First the front view image is copied, then the outer garment is removed for Outer scans
# for Inner scans it's just copied as {scan_name}_outer.png since it doesn't have outer garment
# then the inner garment is removed and the generated image is saved as {scan_name}_inner.png
# lastly, the lower garment is removed and the generated image is saved as {scan_name}_lower.png
def main(dataset_dir, garment_data_json, index):
    with open(garment_data_json, 'r') as f:
        garment_data = json.load(f)

    scan_names = list(garment_data.keys())
    scan_name = scan_names[index-1]
    scan_dict = garment_data[scan_name]
    scan_gen_args = scan_dict['flux_kontext_args']
    print(f"Processing scan {scan_name} with index {index-1}.")

    copy_filename = f"./test_fkon/{scan_name}.png"
    if not os.path.isfile(copy_filename):
        scan_dir = os.path.join(dataset_dir, scan_name)
        initial_anchor_idx = scan_dict['anchor_idx']

        image_path = os.path.join(scan_dir, 'images', f'train_{initial_anchor_idx:04d}.png')
        scan_image = transp_to_white(Image.open(image_path))
        scan_image.save(copy_filename)

    # Load FluxKontext
    pipe_kontext = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16, safety_checker=None).to("cuda")
    pipe_kontext.safety_checker = disabled_safety_checker
    outer_filename = f"./test_fkon/{scan_name}_1_outer.png"
    if not os.path.isfile(outer_filename):
        scan_image = Image.open(copy_filename)
        if 'outer' not in scan_gen_args:
            scan_image.save(outer_filename)
        else:
            prompt = scan_gen_args['outer']['prompt']
            seed = scan_gen_args['outer']['seed']
            seed = random.randint(0, MAX_SEED) if seed == -1 else seed
            print(f'Will remove outer garment for {scan_name} with prompt {prompt} and seed {seed}.')
            gen_image = remove_garment_kontext(pipe_kontext, scan_image, prompt, seed=seed)
            gen_image.save(outer_filename)
            print(f"Generated image saved as {outer_filename}")
            del gen_image; gc.collect(); torch.cuda.empty_cache()
        
    inner_filename = f"./test_fkon/{scan_name}_2_inner.png"
    if not os.path.isfile(inner_filename):
        image_no_outer = Image.open(outer_filename)
        prompt = scan_gen_args['inner']['prompt']
        neg_prompt = scan_gen_args['inner']['negative_prompt'] if 'negative_prompt' in scan_gen_args['inner'] else None
        true_cfg_scale = scan_gen_args['inner']['true_cfg_scale'] if 'true_cfg_scale' in scan_gen_args['inner'] else 1.0
        num_inference_steps = scan_gen_args['inner']['num_inference_steps'] if 'num_inference_steps' in scan_gen_args['inner'] else 28
        guidance_scale = scan_gen_args['inner']['guidance_scale'] if 'guidance_scale' in scan_gen_args['inner'] else 3.5
        seed = scan_gen_args['inner']['seed']
        seed = random.randint(0, MAX_SEED) if seed == -1 else seed
        print(f'Will remove inner garment for {scan_name} with prompt {prompt} and seed {seed}.')
        gen_image = remove_garment_kontext(pipe_kontext, image_no_outer, prompt, negative_prompt=neg_prompt, 
                                            true_cfg_scale=true_cfg_scale, num_inference_steps=num_inference_steps, 
                                            guidance_scale=guidance_scale, seed=seed)
        gen_image.save(inner_filename)
        print(f"Generated image saved as {inner_filename}")
        del gen_image; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <dataset_dir> <garment_data_json> <index>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    garment_data_json = sys.argv[2]
    index = int(sys.argv[3])
    main(dataset_dir, garment_data_json, index)
