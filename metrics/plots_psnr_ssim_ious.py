import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

# --- Configuration ---
JSON_FILE = 'file.json'
OUTPUT_DIR = 'plots_and_tables'

# Mapping for clear labeling in plots and tables
METRIC_MAP = {
    'ssim_inner': 'Inner Garment Avg. SSIM',
    'psnr_inner': 'Inner Garment Avg. PSNR',
    'psnr_nongen_remove_outer': 'PSNR (Unmodified region - Outer Removal)',
    'ssim_nongen_remove_outer': 'SSIM (Unmodified region - Outer Removal)',
    'psnr_nongen_remove_inner': 'PSNR (Unmodified region - Inner Removal)',
    'ssim_nongen_remove_inner': 'SSIM (Unmodified region - Inner Removal)',
    'ious_orig-remove_outer': f"IOU (Outer Removal)",
    'ious_orig-remove_inner': f"IOU (Inner Removal)",
}

def load_and_process_data(file_path):
    """
    Loads data from the JSON file and processes it into a clean pandas DataFrame.
    It unnests the data to have one row per camera index per scan and handles invalid values.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    processed_rows = []
    for scan_id, scan_data in data.items():
        indices = scan_data.get('indices', [])
        if not indices:
            continue

        for i, index_val in enumerate(indices):
            row = {'scan': scan_id, 'index': index_val}
            
            # Process SSIM and PSNR metrics
            for key, name in METRIC_MAP.items():
                if key in scan_data and scan_data[key]:
                    try:
                        val = float(scan_data[key][i])
                        row[name] = val if val != -1.0 else np.nan
                    except (IndexError, TypeError):
                        row[name] = np.nan
                else:
                    row[name] = np.nan
            processed_rows.append(row)

    df = pd.DataFrame(processed_rows)
    # Reorder columns for logical presentation
    base_metrics = list(METRIC_MAP.values())
    df = df[['scan', 'index'] + base_metrics]
    
    return df

def analyze_and_plot(df, group_by_col, analysis_name, generate_bar_plots=True):
    """
    Analyzes data by grouping, prints a table, generates bar plots, and returns the averaged DataFrame.
    """
    print(f"\n--- Averages per {analysis_name} ---\n")
    
    # Calculate mean, ignoring NaN values and non-numeric columns
    avg_df = df.groupby(group_by_col).mean(numeric_only=True)

    # Save and print the table
    table_path = os.path.join(OUTPUT_DIR, f'table_avg_per_{analysis_name}.csv')
    avg_df.round(3).to_csv(table_path)
    print(f"Table of averages per {analysis_name}:")
    print(avg_df.round(3).to_string())
    print(f"\nFull table saved to {table_path}")

    # Generate and save bar plots
    if generate_bar_plots:
        print(f"Generating bar plots for averages per {analysis_name}...")
        for column in avg_df.columns:
            plt.figure(figsize=(10, 6))
            avg_df[column].plot(kind='bar')
            plt.title(f'Average {column} per {analysis_name}')
            plt.xlabel(analysis_name)
            plt.ylabel(f'Average {column}')
            plt.tight_layout()
            
            # Sanitize filename
            safe_col_name = "".join(c for c in column if c.isalnum() or c in (' ', '-', '_')).rstrip()
            bar_path = os.path.join(OUTPUT_DIR, f'bar_avg_{safe_col_name}_per_{analysis_name}.png')
            plt.savefig(bar_path)
            plt.close()
        print("Bar plots saved.")
    
    # *** CHANGE: Return the averaged dataframe for further use ***
    return avg_df


def analyze_overall_metrics(df):
    """
    Calculates and prints the overall average for each metric.
    """
    print("\n--- Overall Averages Across All Scans and Indices ---\n")
    # Drop non-numeric columns for overall mean calculation
    overall_avg = df.drop(columns=['scan', 'index']).mean()
    
    table_path = os.path.join(OUTPUT_DIR, 'table_overall_averages.csv')
    overall_avg.round(4).to_csv(table_path)

    print(overall_avg.round(4).to_string())
    print(f"\nOverall averages table saved to {table_path}")
    

def generate_boxplots(df, suffix):
    """
    Generates box plots for the main metrics from the given DataFrame.
    The suffix is used to create a descriptive title and filename.
    """
    print(f"\n--- Generating Box Plots for {suffix} ---\n")
    main_metrics = list(METRIC_MAP.values())
    
    if not main_metrics:
        print("No main metrics defined, skipping box plots.")
        return

    plt.figure(figsize=(20, 12))
    # Use the DataFrame passed as an argument
    df[main_metrics].plot(kind='box', subplots=True, layout=(2, 4), figsize=(18, 10), sharey=False)
    
    # Update the title to be more descriptive based on the suffix
    plt.suptitle(f'Distribution of Core Metrics ({suffix})', fontsize=18, y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f'boxplot_metric_distributions_{suffix.replace(" ", "_").lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Metric distribution box plots saved to {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plots.py <json_glob_pattern>")
        sys.exit(1)

    json_glob = sys.argv[1]
    json_files = glob.glob(json_glob)
    if not json_files:
        print(f"No JSON files found matching pattern: {json_glob}")
        sys.exit(1)

    for JSON_FILE in json_files:
        if 'sweeping' in JSON_FILE:
            method = os.path.basename(JSON_FILE)[:len('sweeping_anchors_1_1_2_0')]
        elif 'equally' in JSON_FILE:
            method = os.path.basename(JSON_FILE)[:len('equallyspaced_anchors_3_2_3_2_5')]
        else:
            method = os.path.splitext(os.path.basename(JSON_FILE))[0]
        OUTPUT_DIR = f"./plots_psnr_ssim_ious/{method}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load and process data
        full_df = load_and_process_data(JSON_FILE)
        full_df.round(4).to_csv(os.path.join(OUTPUT_DIR, 'full.csv'))
        print(f"Data loaded and processed successfully for {JSON_FILE}.")

        # --- Run Analyses ---

        # 1. Averages per Index (Camera View)
        avg_per_index_df = analyze_and_plot(full_df, 'index', 'Index')
        generate_boxplots(avg_per_index_df, "Per Index Averages")

        # 2. Averages per Scan, and store the resulting table for the box plot
        avg_per_scan_df = analyze_and_plot(full_df, 'scan', 'Scan', generate_bar_plots=False)
        generate_boxplots(avg_per_scan_df, "Per Scan Averages")

        # 3. Overall Averages
        analyze_overall_metrics(full_df)

        print(f"\n✅ Analysis complete. All outputs are saved in the '{OUTPUT_DIR}' directory for {JSON_FILE}.")