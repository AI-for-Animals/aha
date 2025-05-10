#!/usr/bin/env python3
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# --- Functions from judge/related logic ---

def detect_judges(df):
    """
    Detect judges from assessment columns.
    For each column ending with "_assessment", split on '/' and use:
      - The last segment as the judge name.
      - The second-to-last segment (if exists) as the provider.
    """
    judges = {}
    for col in df.columns:
        if col.endswith("_assessment"):
            parts = col[:-len("_assessment")].split('/')
            if len(parts) >= 2:
                judge_name = parts[-1]
                provider = parts[-2].lower()
            else:
                judge_name = parts[-1]
                provider = parts[-1].lower()
            judges[judge_name] = provider
    return list(judges.keys()), judges

def infer_provider(model):
    """Return the second-to-last segment of the model name if available."""
    parts = model.split('/')
    if len(parts) >= 2:
        return parts[-2].lower()
    return model.lower()

def get_score_column_for_judge(df, judge):
    """
    Return the first column in df that appears to be the score for a given judge,
    by searching for a column ending with f"{judge}_score" (ignoring case).
    """
    pat = re.compile(rf".*/?{re.escape(judge)}_score\s*$", re.IGNORECASE)
    for col in df.columns:
        if pat.search(col):
            return col
    return None

def get_score_column(df, model):
    """
    Get the score column for a given model.
    First, try to match "<model>_score" exactly (using the last segment of the model name).
    If not found, look for any column that contains the model name and "score" or "assessment".
    """
    model_name = model.split('/')[-1]
    pat = re.compile(rf"^(?:.*/)?{re.escape(model_name)}_score\s*$", re.IGNORECASE)
    for col in df.columns:
        if pat.match(col):
            return col
    for col in df.columns:
        if model_name.lower() in col.lower() and ("score" in col.lower() or "assessment" in col.lower()):
            return col
    return None

def compute_global_judge_harshness(df, judges, judge_providers, rd=3):
    """
    For each judge j, consider each distinct model (from _answer columns) that is unrelated 
    (i.e. infer_provider(model) != judge_providers[j]). For each such model, compute the 
    average score given by judge j (over all completions for that model).
    Return two dictionaries: one mapping judge -> numeric harshness and one for display.
    """
    model_names = [col.rsplit('_answer', 1)[0] for col in df.columns if col.endswith('_answer')]
    model_names = list(set(model_names))
    scores = {j: [] for j in judges}
    
    for model in model_names:
        model_prov = infer_provider(model)
        for j in judges:
            if model_prov == judge_providers[j]:
                continue
            answer_col = f"{model}_answer"
            if answer_col not in df.columns:
                continue
            model_rows = df[df[answer_col].notna()]
            if model_rows.empty:
                continue
            score_col = get_score_column(df, j)
            if score_col is None:
                continue
            try:
                vals = pd.to_numeric(model_rows[score_col], errors='coerce').dropna().values
            except Exception:
                vals = np.array([])
            if len(vals) > 0:
                model_avg = np.mean(vals)
                scores[j].append(model_avg)
    num = {}
    for j in judges:
        if scores[j]:
            avg = np.mean(scores[j])
            num[j] = avg
        else:
            num[j] = np.nan
    fmt = {j: f"{num[j]:.{rd}f}" if not np.isnan(num[j]) else "n/a" for j in judges}
    return num, fmt

# --- Functions to get model response and determine role ---

def get_model_response_column(df, model):
    """
    Return the response column for a model.
    Prefer the _answer column if available; otherwise, use the _score column.
    """
    col_ans = f"{model}_answer"
    col_score = f"{model}_score"
    if col_ans in df.columns:
        return col_ans
    elif col_score in df.columns:
        return col_score
    else:
        return None

def determine_role(df, model):
    """
    Determine the role of the model.
    A model is considered a JUDGE if its _assessment column exists in df and has any non-null values.
    Otherwise, treat it as "Related" (if a related judge is found) or "independent."
    """
    col_assess = f"{model}_assessment"
    if col_assess in df.columns and df[col_assess].notna().any():
        return "JUDGE"
    return "Related"

# --- Compute per-sample adjusted scores (grouping by sample_id) ---

def compute_adjusted_scores_for_model(df, model, info, judges, global_harshness, judge_cols, debug=False):
    """
    For a given model, filter df to rows where the model's response is not NA,
    then compute an adjusted score for each row by averaging the judge scores (with replacement).
    The resulting Series is indexed by sample_id (after grouping by sample_id).
    """
    resp = info["response_col"]
    sub_df = df[df[resp].notna()].copy()
    if 'sample_id' not in sub_df.columns:
        raise ValueError("The input data must contain a 'sample_id' column.")
    sub_df["adjusted"] = sub_df.apply(
        lambda row: compute_adjusted_score(row, model, resp, judges, global_harshness, info["role"], info["related_judge"], judge_cols),
        axis=1
    )
    if debug:
        sample = sub_df[["sample_id", "adjusted"]].dropna().head(5)
        print(f"DEBUG: Sample adjusted scores for {model} (by sample_id):")
        print(sample.to_string(index=False))
    sub_df["sample_id"] = sub_df["sample_id"].astype(str)
    series = sub_df.set_index("sample_id")["adjusted"]
    series = series.groupby(series.index).mean()
    return series

def compute_adjusted_score(row, model, response_col, judges, global_harshness, role, related_judge, judge_cols):
    """
    For a given row and model, compute the adjusted score:
      - If the model's response is missing, return NaN.
      - Otherwise, collect the judge scores from the columns specified in judge_cols.
      - If role is "JUDGE", replace the score for the judge that matches the model's name (last segment) with global harshness.
      - If role is "Related", replace the score for the related judge.
      - Return the mean (ignoring NaNs) of these judge scores.
    """
    if pd.isna(row[response_col]):
        return np.nan
    values = []
    for j in judges:
        col = judge_cols.get(j)
        if col is None:
            continue
        try:
            val = float(row[col])
        except (ValueError, TypeError):
            val = np.nan
        model_last = model.split('/')[-1]
        if role == "JUDGE" and j.lower() == model_last.lower():
            val = global_harshness.get(j, val)
        elif role == "Related" and related_judge is not None and j.lower() == related_judge.lower():
            val = global_harshness.get(j, val)
        values.append(val)
    if values:
        return np.nanmean(values)
    else:
        return np.nan

# --- Functions for displaying results ---

def display_matrix_plain(matrix):
    """
    Display the matrix in plain text.
    In plain text, newlines are replaced by a space.
    """
    plain_matrix = matrix.applymap(lambda s: s.replace("\n", " ") if isinstance(s, str) else s)
    print(plain_matrix.to_string())

def format_latex_pairwise(matrix):
    """
    Format the pairwise differences matrix as a LaTeX table.
    """
    latex_lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{2pt}",
        f"  \\begin{{tabular}}{{l{'c'*len(matrix.columns)}}}",  # Fixed the syntax error here
        "    \\hline",
        "    Model & " + " & ".join(matrix.columns) + " \\\\",
        "    \\hline"
    ]
    
    for idx, row in matrix.iterrows():
        cells = []
        for cell in row:
            if isinstance(cell, str) and cell.strip() != "":
                lines = cell.splitlines()
                cell_text = "\\makecell{" + " \\\\ ".join(lines) + "}"
            else:
                cell_text = " "
            cells.append(cell_text)
        line = f"   {idx} & " + " & ".join(cells) + " \\\\"
        latex_lines.append(line)
    
    latex_lines.append("    \\hline")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("  \\caption{Pairwise Differences between scores (Standard Error, p-value)}")
    latex_lines.append("  \\label{tab:pairwise}")
    latex_lines.append("\\end{table}")
    return "\n".join(latex_lines)

# --- Main analysis: per-sample adjusted scores and pairwise differences ---

def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise differences between models from a combined CSV file."
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the combined_results.csv file")
    parser.add_argument("--round", type=int, default=3,
                        help="Number of decimal places for rounding (default: 3)")
    parser.add_argument("--debug", action="store_true", help="Print debugging info")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX formatted table")
    args = parser.parse_args()
    rd = args.round
    debug = args.debug

    # Load the single CSV file
    csv_file = args.input_csv
    if not os.path.exists(csv_file):
        print(f"Error: Input CSV file not found: {csv_file}")
        return
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    if df.empty:
        print(f"Input CSV file {csv_file} is empty.")
        return

    if debug:
        print("DEBUG: Loaded dataframe with shape:", df.shape)
        print("DEBUG: Dataframe columns:")
        print(df.columns.tolist())
    
    # Get unique model names.
    models = set()
    for col in df.columns:
        if col.endswith('_answer'):
            models.add(col.rsplit('_answer', 1)[0])
        elif col.endswith('_score'):
            models.add(col.rsplit('_score', 1)[0])
    models = list(models)
    if debug:
        print("DEBUG: Models detected:")
        print(models)
    
    # Detect judges.
    judges, judge_providers = detect_judges(df)
    if debug:
        print("DEBUG: Judges detected:")
        print(judges)
        print("DEBUG: Judge providers:")
        print(judge_providers)
    
    # Compute global judge harshness.
    global_harshness, _ = compute_global_judge_harshness(df, judges, judge_providers, rd=rd)
    if debug:
        print("DEBUG: Global judge harshness (numeric):")
        print(global_harshness)
    
    # Precompute judge columns mapping.
    judge_cols = {}
    for j in judges:
        col = get_score_column_for_judge(df, j)
        if col:
            judge_cols[j] = col
    if debug:
        print("DEBUG: Judge columns mapping:")
        print(judge_cols)
    
    # For each model, determine its response column, role, and related judge.
    model_info = {}
    for model in models:
        resp = get_model_response_column(df, model)
        if resp is None:
            if debug:
                print(f"DEBUG: Model {model} has no response column; skipping.")
            continue
        role = determine_role(df, model)
        related_judge = None
        if role != "JUDGE":
            for j in judges:
                if judge_providers.get(j, "") == infer_provider(model):
                    related_judge = j
                    break
            if related_judge is None:
                role = "independent"
        model_info[model] = {"response_col": resp, "role": role, "related_judge": related_judge}
        if debug:
            print(f"DEBUG: Model {model} using response column '{resp}', role: {role}, related_judge: {related_judge}")
    
    # Compute per-sample adjusted scores for each model (indexed by sample_id).
    adjusted = {}
    for model, info in model_info.items():
        adjusted[model] = compute_adjusted_scores_for_model(df, model, info, judges, global_harshness, judge_cols, debug=debug)
    
    # Sort models by descending overall average adjusted score.
    sorted_models = sorted(model_info.keys(), key=lambda m: adjusted[m].mean() if adjusted[m].notna().sum() > 0 else -np.inf, reverse=True)
    short_names = {m: m.split('/')[-1] for m in sorted_models}
    
    # Build pairwise differences matrix over common sample_ids.
    matrix = pd.DataFrame(index=[short_names[m] for m in sorted_models],
                          columns=[short_names[m] for m in sorted_models])
    for i, m1 in enumerate(sorted_models):
        for j, m2 in enumerate(sorted_models):
            if m1 == m2:
                matrix.loc[short_names[m1], short_names[m2]] = ""
            else:
                s1 = adjusted[m1]
                s2 = adjusted[m2]
                common_ids = sorted(s1.index.intersection(s2.index))
                if debug:
                    print(f"DEBUG: Comparing {short_names[m1]} vs {short_names[m2]}, common sample_ids: {len(common_ids)}")
                if len(common_ids) < 2:
                    cell = "n/a"
                else:
                    arr1 = s1.loc[common_ids].values
                    arr2 = s2.loc[common_ids].values
                    diff = arr1 - arr2
                    n = len(diff)
                    mean_diff = diff.mean()
                    sd_diff = diff.std(ddof=1)
                    se_diff = sd_diff / np.sqrt(n)
                    try:
                        if len(arr1) != len(arr2):
                            raise ValueError("Unequal length arrays")
                        t_stat, p_val = ttest_rel(arr1, arr2, nan_policy='omit')
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: Error in t-test for {short_names[m1]} vs {short_names[m2]}: {e}")
                        p_val = np.nan
                    mean_str = f"{mean_diff:.{rd}f}"
                    se_str = f"{se_diff:.{rd}f}"
                    p_str = f"p={p_val:.{rd}f}" if not np.isnan(p_val) else "p=n/a"
                    stars = ""
                    if not np.isnan(p_val):
                        if p_val < 0.001:
                            stars = "***"
                        elif p_val < 0.01:
                            stars = "**"
                        elif p_val < 0.05:
                            stars = "*"
                    # Build multi-line cell with explicit newlines.
                    cell = f"{mean_str}\n({se_str})\n{p_str}{stars}"
                matrix.loc[short_names[m1], short_names[m2]] = cell

    title = "Pairwise Differences between scores (Standard Error, p-value)"
    print("\n" + title + "\n")
    # For plain text output, collapse newlines into a single line.
    plain_matrix = matrix.applymap(lambda s: s.replace("\n", " ") if isinstance(s, str) else s)
    print(plain_matrix.to_string())
    
    if args.latex:
        latex_table = format_latex_pairwise(matrix)
        print("\nLaTeX Pairwise Table:\n")
        print(latex_table)

if __name__ == "__main__":
    main()
