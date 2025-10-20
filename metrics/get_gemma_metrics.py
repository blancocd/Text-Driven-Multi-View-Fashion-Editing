import argparse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
import os
from tqdm import tqdm
import cv2
import numpy as np
import re
def load_mask(filename):
    img = cv2.imread(filename)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(bool)
    return mask

def main(gen_method_dir, garment_data_json, indices_step_sample):
    with open(garment_data_json, 'r') as f:
        garment_data = json.load(f)
    scan_names = list(garment_data.keys())
    
    method = os.path.basename(os.path.normpath(gen_method_dir))
    print(f"Evaluating results from {method} method.")

    model_id = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    results_dict = {}
    for scan_name in tqdm(scan_names):
        garment_dict = garment_data[scan_name]["flux_fill_args"]
        img_fns = sorted([f for f in os.listdir(os.path.join(gen_method_dir, scan_name, 'inner', 'images')) if f.startswith('train')])
        gen_indices = [int(img_fn.split('_')[1].split('.')[0]) for img_fn in img_fns]
        selected_indices = list(range(0, len(img_fns), indices_step_sample))

        results_dict[scan_name] = {
            'indices': [gen_indices[i] for i in selected_indices],
            'outer': {
                'succesfully_removed': [],
                'succesfully_removed_full_answer': [],
                'removal_quality': [],
                'removal_quality_full_answer': []
            },
            'inner': {
                'succesfully_removed': [],
                'succesfully_removed_full_answer': [],
                'removal_quality': [],
                'removal_quality_full_answer': []
            }
        }

        for img_fn in [img_fns[i] for i in selected_indices]:
            gen_scan_dir = os.path.join(gen_method_dir, scan_name)
            
            # Rating outer garment removal
            if os.path.isdir(os.path.join(gen_scan_dir, 'outer')):
                flux_fill_outer_prompt = garment_dict["outer"]["prompt"]
                flux_fill_outer_prompt = flux_fill_outer_prompt[2:] if flux_fill_outer_prompt[:2] == 'a ' else flux_fill_outer_prompt
                gen_remove_outer_img_path = os.path.join(gen_scan_dir, 'outer', 'images', img_fn)
                succesfully_removed_outer_prompt = (
                    f'Focusing only on the upper body garment and not the shoes or pants:\n'
                    f'Answer yes if the person in the image is only wearing a {flux_fill_outer_prompt} '
                    f'Answer no if the person is wearing a jacket, blazer, or outer garment on top of the {flux_fill_outer_prompt}.'
                )
                outer_removal_quality_prompt = (
                    f'The person in the image had their jacket, blazer, or outer garment removed and thus should only be wearing a {flux_fill_outer_prompt}. '
                    f'Your task is to rate the quality of the removal. '
                    f'On a scale from 1 to 10, rate the removal. '
                    f'10 means that the removal was successful and the person is only wearing a {flux_fill_outer_prompt}. '
                    f'1 means that the removal was unsuccessful and the person is wearing a jacket, blazer, or outer garment on top of the {flux_fill_outer_prompt}.'
                )

                # First prompt: succesfully_removed
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant focused on describing and answering questions about the clothing worn by humans."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": gen_remove_outer_img_path},
                            {"type": "text", "text": succesfully_removed_outer_prompt}
                        ]
                    }
                ]

                inputs_outer_removal_evaluation = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    generation = model.generate(**inputs_outer_removal_evaluation, max_new_tokens=500, do_sample=False)
                    generation = generation[0][inputs_outer_removal_evaluation["input_ids"].shape[-1]:]

                outer_removal_answer_decoded = processor.decode(generation, skip_special_tokens=True)
                results_dict[scan_name]['outer']["succesfully_removed"].append('yes' in outer_removal_answer_decoded.lower())
                results_dict[scan_name]['outer']["succesfully_removed_full_answer"].append(outer_removal_answer_decoded)

                # Second prompt: removal_quality
                messages_quality = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant focused on describing and answering questions about the clothing worn by humans."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": gen_remove_outer_img_path},
                            {"type": "text", "text": succesfully_removed_outer_prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": outer_removal_answer_decoded}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": outer_removal_quality_prompt}]
                    }
                ]

                inputs_inner_removal_evaluation = processor.apply_chat_template(
                    messages_quality, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    generation_quality = model.generate(**inputs_inner_removal_evaluation, max_new_tokens=500, do_sample=False)
                    generation_quality = generation_quality[0][inputs_inner_removal_evaluation["input_ids"].shape[-1]:]

                outer_removal_quality_decoded = processor.decode(generation_quality, skip_special_tokens=True)
                match = re.search(r'\d+', outer_removal_quality_decoded)
                if match:
                    results_dict[scan_name]['outer']["removal_quality"].append(int(match.group()))
                else:
                    results_dict[scan_name]['outer']["removal_quality"].append(-1)
                results_dict[scan_name]['outer']["removal_quality_full_answer"].append(outer_removal_quality_decoded)

            # Rating inner garment removal
            gen_remove_inner_img_path = os.path.join(gen_scan_dir, 'inner', 'images', img_fn)
            flux_fill_inner_prompt = garment_dict["inner"]["prompt"]
            if flux_fill_inner_prompt == "a man's torso":
                description_upper_body = "man showing his torso"
                initial_prompt = f'Focusing only on the upper body garment and not the shoes or pants:\n'
            elif flux_fill_inner_prompt == "sports bra":
                description_upper_body = "woman wearing a sports bra"
                initial_prompt = f'Focusing only on the upper body garment and not the shoes or pants:\n'
            elif flux_fill_inner_prompt == "sports bra and shorts":
                description_upper_body = "woman wearing a sports bra and shorts"
                initial_prompt = f'Focusing only on the garments and not the shoes:\n'
            else:
                print(f"Flux Fill prompt is unexpected: {flux_fill_inner_prompt}")

            if "outer" in garment_dict:
                flux_fill_outer_prompt = flux_fill_outer_prompt[2:] if flux_fill_outer_prompt[:2] == 'a ' else flux_fill_outer_prompt
                description_outer_body = f"{flux_fill_outer_prompt},"
            else:
                description_outer_body = f""
            succesfully_removed_inner_prompt = (
                f'{initial_prompt}'
                f'Answer yes if the person in the image is a {description_upper_body}. '
                f'Answer no if the person is wearing a {description_outer_body} shirt or upper body clothing.'
            )
            inner_removal_quality_prompt = (
                f'The person in the image had their {description_outer_body} shirt, or upper body clothing removed and thus should be a {description_upper_body}. '
                f'Your task is to rate the quality of the removal. '
                f'On a scale from 1 to 10, rate the removal. '
                f'10 means that the removal was successful and the person is a {description_upper_body}. '
                f'0 means that the removal was unsuccessful and the person is wearing a {description_outer_body} shirt, or upper body clothing.'
            )

            # First prompt: succesfully_removed
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant focused on describing and answering questions about the clothing worn by humans."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": gen_remove_inner_img_path},
                        {"type": "text", "text": succesfully_removed_inner_prompt}
                    ]
                }
            ]

            inputs_inner_removal_evaluation = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                generation = model.generate(**inputs_inner_removal_evaluation, max_new_tokens=500, do_sample=False)
                generation = generation[0][inputs_inner_removal_evaluation["input_ids"].shape[-1]:]

            inner_removal_answer_decoded = processor.decode(generation, skip_special_tokens=True)
            results_dict[scan_name]['inner']["succesfully_removed"].append('yes' in inner_removal_answer_decoded.lower())
            results_dict[scan_name]['inner']["succesfully_removed_full_answer"].append(inner_removal_answer_decoded)

            # Second prompt: removal_quality
            messages_quality = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant focused on describing and answering questions about the clothing worn by humans."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": gen_remove_inner_img_path},
                        {"type": "text", "text": succesfully_removed_inner_prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": inner_removal_answer_decoded}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": inner_removal_quality_prompt}]
                }
            ]

            inputs_inner_removal_evaluation = processor.apply_chat_template(
                messages_quality, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                generation_quality = model.generate(**inputs_inner_removal_evaluation, max_new_tokens=500, do_sample=False)
                generation_quality = generation_quality[0][inputs_inner_removal_evaluation["input_ids"].shape[-1]:]

            inner_removal_quality_decoded = processor.decode(generation_quality, skip_special_tokens=True)
            match = re.search(r'\d+', inner_removal_quality_decoded)
            if match:
                results_dict[scan_name]['inner']["removal_quality"].append(int(match.group()))
            else:
                results_dict[scan_name]['inner']["removal_quality"].append(-1)
            results_dict[scan_name]['inner']["removal_quality_full_answer"].append(inner_removal_quality_decoded)
                
    results_fn = f'{method}_{indices_step_sample}_gemma_results.json'
    with open(results_fn, 'w') as f:
        json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate inpainting results.")
    parser.add_argument("--gen_method_dir", type=str, help="Directories with result images")
    parser.add_argument("--garment_data_json", type=str, help="JSON with garment captions")
    parser.add_argument("--indices_step_sample", type=int, help="Sampling step of indices to evaluate per scan")
    args = parser.parse_args()
    main(args.gen_method_dir, args.garment_data_json, args.indices_step_sample)
