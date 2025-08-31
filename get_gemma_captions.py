import argparse
import os
import json
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def generate_caption(model, processor, image_path, prompt):
    """
    Generates a caption for a single image.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process the inputs
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    # Generate the caption
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        # Slice to get only the generated text
        generation = generation[0][inputs["input_ids"].shape[-1]:]
    
    # Decode the caption
    caption = processor.decode(generation, skip_special_tokens=True)
    return caption.strip()

def main(image_dir, output_file):
    """
    Main function to find images, generate captions, and save to JSON.
    """
    print("Starting the image captioning process...")

    # --- 1. Load Model and Processor ---
    # Using the model from your example, suitable for vision-language tasks.
    model_id = "google/gemma-3-4b-it"
    print(f"Loading model: {model_id}")
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have run 'huggingface-cli login' and have access to the model.")
        return

    # --- 2. Find and Group Image Files ---
    print(f"Scanning for images in: {image_dir}")
    image_groups = defaultdict(dict)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith('_inner.png'):
            base_name = filename[:-12] # Remove '_inner.png'
            image_groups[base_name]['inner'] = os.path.join(image_dir, filename)
        elif filename.lower().endswith('_outer.png'):
            base_name = filename[:-12] # Remove '_outer.png'
            image_groups[base_name]['outer'] = os.path.join(image_dir, filename)

    if not image_groups:
        print("No images matching the '_inner.png' or '_outer.png' pattern were found.")
        return
        
    print(f"Found {len(image_groups)} groups of images to process.")

    # --- 3. Generate Captions for Each Image ---
    final_captions = {}
    prompt = "Provide a concise, one-sentence caption for this image."

    for base_name, paths in tqdm(image_groups.items(), desc="Captioning Images"):
        final_captions[base_name] = {}
        for key, image_path in paths.items(): # key is 'inner' or 'outer'
            try:
                caption = generate_caption(model, processor, image_path, prompt)
                final_captions[base_name][key] = caption
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                final_captions[base_name][key] = f"Error: Failed to generate caption."

    # --- 4. Save Results to JSON ---
    print(f"Saving captions to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(final_captions, f, indent=4)
    
    print("Captioning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images using Gemma.")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Directory containing the images to be captioned."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="captions.json", 
        help="Path to the output JSON file."
    )
    args = parser.parse_args()
    main(args.image_dir, args.output_file)