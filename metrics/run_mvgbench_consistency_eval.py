import re
import os
import json
import shutil
import argparse
import subprocess

def prepare_data(dataset_dir, gen_dir, scan_name, method):
    transforms_path = os.path.join(dataset_dir, scan_name, "transforms_train.json")

    for removal_type in ['inner', 'outer']:
        with open(transforms_path, "r") as f:
            transforms = json.load(f)

        gen_scan_dir = os.path.join(gen_dir, scan_name, removal_type)
        print(f"Preparing data for ", gen_scan_dir)
        if not os.path.exists(gen_scan_dir):
            continue
        odd_dir = os.path.join(gen_dir, f"odd_views_{method}", f'{scan_name}_{removal_type}')
        even_dir = os.path.join(gen_dir, f"even_views_{method}", f'{scan_name}_{removal_type}')
        os.makedirs(os.path.join(odd_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(even_dir, "images"), exist_ok=True)

        # Separate frames into odd and even lists
        gen_frames = []
        for frame in transforms["frames"]:
            # Extract the image number from the file path
            num_str = os.path.basename(frame["image_path"]).split("_")[-1]
            img_num_transforms = int(num_str)

            # Get the generated image path
            original_img_path = os.path.join(gen_dir, scan_name, removal_type, "images",
                                             f"train_{img_num_transforms:04d}.png")
            if not os.path.exists(original_img_path):
                continue
            
            frame["file_path"] = f"./images/train_{img_num_transforms:04d}"
            gen_frames.append(frame)

        for frame in gen_frames:
            # Copy and rename the image
            num_str = os.path.basename(frame["file_path"]).split("_")[-1]
            img_num_transforms = int(num_str)
            original_img_path = os.path.join(gen_dir, scan_name, removal_type, "images",
                                             f"train_{img_num_transforms:04d}.png")
            
            for d in [even_dir, odd_dir]:
                new_img_path = os.path.join(d, frame["file_path"] + '.png')
                shutil.copy(original_img_path, new_img_path)

        # Create new transforms for odd and even views
        even_transforms = {**transforms, "frames": gen_frames[::2]}
        odd_transforms = {**transforms, "frames": gen_frames[1::2]}

        # Write the new transforms to JSON files
        with open(os.path.join(odd_dir, "transforms_train.json"), "w") as f:
            json.dump(odd_transforms, f, indent=4)
        with open(os.path.join(even_dir, "transforms_train.json"), "w") as f:
            json.dump(even_transforms, f, indent=4)
        with open(os.path.join(odd_dir, "transforms_test.json"), "w") as f:
            json.dump(even_transforms, f, indent=4)
        with open(os.path.join(even_dir, "transforms_test.json"), "w") as f:
            json.dump(odd_transforms, f, indent=4)

        print("Data preparation complete.", flush=True)

def run_mvfit(mvg_bench_dir,  gen_dir, scan_name, method):
    print("Running 3DGS fitting for both odd and even views", flush=True)

    for removal_type in ['inner', 'outer']:
        odd_dir = os.path.join(gen_dir, f"odd_views_{method}", f'{scan_name}_{removal_type}')
        even_dir = os.path.join(gen_dir, f"even_views_{method}", f'{scan_name}_{removal_type}')
        if not os.path.exists(odd_dir):
            continue
        
        # Run mvfit for even views
        subprocess.run([
            "python", os.path.join(mvg_bench_dir, "run_mvfit.py"),
            even_dir, "--white_background", "-debug"
        ], check=True, cwd=mvg_bench_dir)

        # Run mvfit for odd views
        subprocess.run([
            "python", os.path.join(mvg_bench_dir, "run_mvfit.py"),
            odd_dir, "--white_background", "-debug"
        ], check=True, cwd=mvg_bench_dir)

    print("3DGS fitting complete.", flush=True)

def run_evaluation(mvg_bench_dir, method):
    print("Running 3D consistency evaluation...", flush=True)

    odd_output_name = f"output/consistency/odd_views_{method}/*"
    even_output_name = f"output/consistency/even_views_{method}/*"
    subprocess.run([
        "python", os.path.join(mvg_bench_dir, "eval", "eval_consistency.py"),
        "--name_odd", odd_output_name,
        "--name_even", even_output_name
    ], check=True, cwd=mvg_bench_dir)

    print("3D consistency evaluation complete.", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Run MVGBench's 3D consistency evaluation pipeline."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="The directory to the 4ddress dataset containing all the scan subdirectories."
    )
    parser.add_argument(
        "--gen_dir", type=str, required=True,
        help="The directory containing the output of the mv generation."
    )
    parser.add_argument(
        "--scan_index", type=int, required=True,
        help="The index of the scan to process (for array jobs)."
    )
    parser.add_argument(
        "--garment_data_json", type=str, required=True,
        help="JSON with prompts per scan."
    )
    parser.add_argument(
        "--mvg_bench_dir", type=str, required=True,
        help="The path to the MVGBench repository."
    )
    parser.add_argument(
        "--skip_data_prep", action="store_true",
        help="Skip the data preparation step."
    )
    parser.add_argument(
        "--skip_mvfit", action="store_true",
        help="Skip the 3DGS fitting step."
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip the 3D consistency evaluation step."
    )
    args = parser.parse_args()

    with open(args.garment_data_json, 'r') as f:
        garment_data = json.load(f)
    scan_names = list(garment_data.keys())
    scan_name = scan_names[args.scan_index]
    print(f'Running MVGBench for scan {scan_name}', flush=True)

    method = os.path.basename(os.path.normpath(args.gen_dir))

    # Run the pipeline
    if not args.skip_data_prep:
        prepare_data(args.data_dir, args.gen_dir, scan_name, method)
    if not args.skip_mvfit:
        run_mvfit(args.mvg_bench_dir, args.gen_dir, scan_name, method)
    if not args.skip_eval:
        if args.scan_index != 0:
            return
        run_evaluation(args.mvg_bench_dir, method)

if __name__ == "__main__":
    main()
