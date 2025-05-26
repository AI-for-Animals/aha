#!/usr/bin/env python3
import os
import glob
import pandas as pd

def combine_csv_files(files):
    """Combine multiple CSV files into a single DataFrame."""
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def combine_csv_results(directory):
    """Combine all results_*.csv files in the directory into a single combined_results.csv file."""
    cfiles = sorted(glob.glob(os.path.join(directory, "results_*.csv")))
    if not cfiles:
        print("No CSV files found to combine.")
        return None
    
    print(f"Found {len(cfiles)} CSV files to combine.")
    df = combine_csv_files(cfiles)
    
    if df.empty:
        print("No data found in CSV files.")
        return None
    
    # Organize columns in a logical order
    cols = list(df.columns)
    answer_cols = [c for c in cols if c.endswith("_answer")]
    tag_cols = [c for c in cols if c.startswith("tag")]
    other_cols = [c for c in cols if c not in answer_cols+tag_cols]
    
    # Put important columns at the front
    front_cols = []
    for c in ["sample_id", "input"]:
        if c in other_cols:
            front_cols.append(c)
            other_cols.remove(c)
    
    # Final column order
    order = front_cols + answer_cols + other_cols + tag_cols
    df = df[order]
    
    # Save the combined file
    output_path = os.path.join(directory, "combined_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Combine multiple result CSV files into a single file")
    p.add_argument("--input_directory", default="./results", 
                   help="Directory containing results_*.csv files")
    args = p.parse_args()
    
    combine_csv_results(args.input_directory)
    