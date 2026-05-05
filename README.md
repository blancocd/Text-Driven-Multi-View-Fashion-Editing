# **Text-Driven Multi-View Fashion Editing**

A project for using FluxKontext and FluxFill to remove outer garments in a consistent way from multi-view images of people.

## **1. Installation**

It is highly recommended to use a `conda` environment to manage the dependencies for this project.

First, create and activate a new `conda` environment. We require Python 3.10 or newer:

```bash
conda create -n flux_garment_remover python=3.10
conda activate flux_garment_remover
```

Next, install the required libraries. You **must** install PyTorch and Diffusers using the specific commands below before installing the project requirements:

```bash
# 1. Install PyTorch, torchvision, and torchaudio for CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install the latest version of Diffusers directly from GitHub
pip install git+https://github.com/huggingface/diffusers.git

# 3. Install this project
pip install .
```

## **1.1 Hugging Face**
FluxFill and FluxKontext are gated models. You will need to create an account, get access to them, create a token (and keep it safe and secret), and put it in the scripts `script_kontext_test.sh` and `script_mv.sh`.

## **2. How to Run**

This project requires a powerful GPU (e.g., NVIDIA A100) to run efficiently. We provide instructions for running the scripts on a Slurm cluster.

### **Step 1: Test Your Prompt and Seed (Crucial!)**

Before processing an entire multi-view scan, it is essential to find a good combination of a prompt and a seed for the FluxKontext model. The quality of the initial generation determines the quality of all subsequent views.

The `test_kontext_generation.py` script is designed for this purpose.

**We recommend the following workflow:**

1. Start with a descriptive prompt.
2. Run the test script with several different seeds. A good prompt might only work well with a specific seed.
4. If you don't get good results, try refining the prompt and repeat the process.

To run the test script, you can edit and use the provided shell script `script_kontext_test.sh` which already submits many jobs in an array job. If you set the seed argument to `None` in the Python script, a random seed will be used, and the output image will be saved as `test_{seed}.png`. This helps you identify which random seed produced a desirable result.

### **Step 2: Run the Garment Removal Script**

Once you have a satisfactory prompt and seed, you can run the main script, `remove_garment_mv.py`, to process the entire scan.

The core function to call is `remove_garment_anchors`. Below is an example of how to configure and run it:

```python
import os
from remove_garment_mv import remove_garment_anchors, get_equally_spaced_anchors_indices, get_sweeping_anchors_indices

# --- Configuration ---
scan_dir = 'your_scan_dir'
garment_type = 'upper'
prompt_flux_kontext = 'remove the outer garment'
prompt_flux_fill = 'white long sleeve shirt'
seed_flux_kontext = 0
seed_flux_fill = 0

img_dir = os.path.join(scan_dir, 'images')
num_views = len([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])

# --- Choose an Anchor Strategy ---

# Strategy 1: Equally Spaced Anchors
initial_anchor_idx = 0
num_anchors = 4
indices_list, indices_to_gen_save_flag_list = get_equally_spaced_anchors_indices(initial_anchor_idx, num_views, num_anchors)

# Strategy 2: Sweeping Anchors
initial_anchor_idx = 0
indices_list, indices_to_gen_save_flag_list = get_sweeping_anchors_indices(initial_anchor_idx, num_views)

# --- Run the main function ---
remove_garment_anchors(
  scan_dir, garment_type, prompt_flux_kontext, prompt_flux_fill,
  initial_anchor_idx, indices_list, indices_to_gen_save_flag_list,
  seed_flux_kontext=seed_flux_kontext, seed_flux_fill=seed_flux_fill, verbose=True
)
```

To run this on a Slurm cluster, please modify and use the `script_mv.sh` file.

### **Argument Explanation**

- `scan_dir`: The path to the root directory of the scan. This directory must contain two subdirectories:
  - `images/`: Contains the input images named `train_####.png`.
  - `segmentation_masks/`: Contains the corresponding segmentation masks, also named `train_####.png`.
  - The generated images will be saved in both directories as `gen_####.png`.
- `garment_type`: Specifies the garment to be targeted. Options are:
  - `'upper'`: Includes both inner and outer upper-body garments.
  - `'outer'`: The outermost upper-body garment.
  - `'lower'`: The lower-body garment.
  - These definitions follow the conventions of the 4Ddress paper.
- `prompt_flux_kontext`: The natural language instruction for FluxKontext. This should be a command, e.g., `'remove the jacket'` or `'add a scarf'`.
- `prompt_flux_fill`: The descriptive prompt for the FluxFill inpainting model. This should describe the desired appearance of the inpainted region, not an instruction. We recommend keeping this description concise and consistent with the expected output from FluxKontext. For example: `'blue t-shirt'`.
- `seed_flux_kontext` / `seed_flux_fill`: Integer seeds for the models to ensure reproducibility. As mentioned, the choice of `seed_flux_kontext` can significantly impact the output quality.
- `initial_anchor_idx`: The index of the first `train_####.png` image to be processed by FluxKontext. The result from this view serves as the foundation for generating all other views. **It is highly recommended to select an index where the inner garment (what will be revealed) is most visible.**
- **Anchor Strategies**:
  - `get_equally_spaced_anchors_indices`: This method selects a few "anchor" views distributed evenly around the person. These anchors are generated first, and then the views in between them are filled in. `num_anchors` controls how many such primary views are used.
  - `get_sweeping_anchors_indices`: This method starts from the `initial_anchor_idx` and generates views sequentially in both rotational directions (e.g., `0 -> 1 -> 2...` and `0 -> N-1 -> N-2...`).

## **3. Running on a Slurm Cluster**

Due to the high computational requirements (NVIDIA A100 recommended), we provide scripts to submit jobs to a Slurm-managed cluster.

- **To test prompts and seeds:** Modify and run `script_kontext_test.sh`.
  ```bash
  sbatch script_kontext_test.sh
  ```
- **To run the full pipeline:** Modify and run `script_mv.sh`.
  ```bash
  sbatch script_mv.sh
  ```
