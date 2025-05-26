import argparse
import pandas as pd
from tabulate import tabulate
import os
import glob

def compute_global_stats(df: pd.DataFrame):
    """
    Compute overall statistics from the combined dataframe.
    Returns:
      (number of model answer columns, number of judge assessment columns,
       mean score, score distribution, category distribution, total score count)
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    cat_cols   = [c for c in df.columns if c.endswith("_category")]
    model_cols = [c for c in df.columns if c.endswith("_answer")]
    judge_cols = [c for c in df.columns if c.endswith("_assessment")]

    total_score = 0.0
    total_count = 0
    score_dist  = {'-1': 0, '0': 0, '1': 0}
    cat_dist    = {}

    for idx in df.index:
        for sc_col in score_cols:
            val = df.at[idx, sc_col]
            if pd.notna(val) and val in (-1, 0, 1):
                total_score += val
                total_count += 1
                score_dist[str(int(val))] += 1
        for c_col in cat_cols:
            c_val = str(df.at[idx, c_col]).strip()
            if c_val in {'A', 'B', 'C', 'D', 'E'}:
                cat_dist[c_val] = cat_dist.get(c_val, 0) + 1

    mean_val = total_score / total_count if total_count > 0 else 0.0
    ordered_cat = {k: cat_dist.get(k, 0) for k in sorted(['A','B','C','D','E'])}
    return len(model_cols), len(judge_cols), mean_val, score_dist, ordered_cat, total_count

def compute_entity_stats(df: pd.DataFrame, suffix: str) -> dict:
    """
    For each entity (model or judge) determined by a column suffix,
    compute the number of questions answered, the score distribution, and category counts.
    """
    out = {}
    entity_cols = [c for c in df.columns if c.endswith(suffix)]
    is_judge = (suffix == "_assessment")
    
    for col in entity_cols:
        name = col[:-len(suffix)]
        row_mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        relevant_indices = df.index[row_mask]
        
        q_count = len(relevant_indices)
        sc_count = 0
        sc_sum = 0.0
        sc_dist = {'-1': 0, '0': 0, '1': 0}
        c_dist = {}

        for rid in relevant_indices:
            if is_judge:
                score_col = f"{name}_score"
                cat_col   = f"{name}_category"
                if score_col in df.columns:
                    val = df.at[rid, score_col]
                    if pd.notna(val) and val in (-1, 0, 1):
                        sc_sum += val
                        sc_count += 1
                        sc_dist[str(int(val))] += 1
                if cat_col in df.columns:
                    cat_val = str(df.at[rid, cat_col]).strip()
                    if cat_val in {'A','B','C','D','E'}:
                        c_dist[cat_val] = c_dist.get(cat_val, 0) + 1
            else:
                judge_names = [c[:-11] for c in df.columns if c.endswith("_assessment")]
                for j in judge_names:
                    score_col = f"{j}_score"
                    cat_col = f"{j}_category"
                    if score_col in df.columns:
                        val = df.at[rid, score_col]
                        if pd.notna(val) and val in (-1, 0, 1):
                            sc_sum += val
                            sc_count += 1
                            sc_dist[str(int(val))] += 1
                    if cat_col in df.columns:
                        cat_val = str(df.at[rid, cat_col]).strip()
                        if cat_val in {'A','B','C','D','E'}:
                            c_dist[cat_val] = c_dist.get(cat_val, 0) + 1

        ordered_c_dist = {k: c_dist.get(k, 0) for k in sorted(['A','B','C','D','E'])}
        out[name] = {
            "questions": q_count,
            "score_count": sc_count,
            "sum_score": sc_sum,
            "score_dist": sc_dist,
            "cat_dist": ordered_c_dist
        }
    return out

def stats_to_row(name: str, stats: dict, format_type='console') -> list:
    """Convert stats dictionary to a row for the table with percentages"""
    total_scores = stats["score_count"]
    total_cats = sum(stats["cat_dist"].values())
    
    if format_type == 'console':
        return [
            name,
            stats["questions"],
            stats["score_count"],
            f"{stats['sum_score'] / stats['score_count']:.3f}" if stats['score_count'] > 0 else "0.000",
            f"{(stats['score_dist']['-1'] / total_scores * 100):.1f}" if total_scores > 0 else "0.0",
            f"{(stats['score_dist']['0'] / total_scores * 100):.1f}" if total_scores > 0 else "0.0",
            f"{(stats['score_dist']['1'] / total_scores * 100):.1f}" if total_scores > 0 else "0.0",
            f"{(stats['cat_dist']['A'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['B'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['C'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['D'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['E'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0"
        ]
    else:  # latex
        avg_score = stats['sum_score'] / stats['score_count'] if stats['score_count'] > 0 else 0.000
        return [
            name.replace('_', '\\_'),  # Escape underscores for LaTeX
            f"{stats['questions']:,}",
            f"{stats['score_count']:,}",
            f"{avg_score:.3f}",
            f"{(stats['score_dist']['-1'] / total_scores * 100):.1f}" if total_scores > 0 else "0.0",
            f"{(stats['score_dist']['0'] / total_scores * 100):.1f}" if total_scores > 0 else "0.0",
            f"{(stats['score_dist']['1'] / total_scores * 100):.1f}" if total_scores > 0 else "0.0",
            f"{(stats['cat_dist']['A'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['B'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['C'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['D'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0",
            f"{(stats['cat_dist']['E'] / total_cats * 100):.1f}" if total_cats > 0 else "0.0"
        ]

def format_latex_table(rows, caption, label):
    headers = [
        "Model", "Questions", "Scores", "Avg Score",
        "-1 (\\%)", "0 (\\%)", "1 (\\%)",
        "A (\\%)", "B (\\%)", "C (\\%)", "D (\\%)", "E (\\%)"
    ]
    
    latex = [
        "\\begin{table}[ht]",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\small",
        "    \\centering",
        "    \\begin{tabular}{l|cccccccccccc}",
        "        \\hline",
        "        \\textbf{" + "} & \\textbf{".join(headers) + "} \\\\",
        "        \\hline"
    ]
    
    # Add data rows
    for row in rows[:-1]:  # Exclude the total row for now
        latex.append("        " + " & ".join(str(x) for x in row) + " \\\\")
    
    # Add total row with hline
    latex.extend([
        "        \\hline",
        "        " + " & ".join(str(x) for x in rows[-1]) + " \\\\",
        "        \\hline",
        "    \\end{tabular}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

def main():
    parser = argparse.ArgumentParser("Raw Results Summary Analysis")
    parser.add_argument("--input_csv", required=True,
                        help="Path to the combined_results.csv file")
    parser.add_argument("--latex", action="store_true",
                        help="If set, output the summary table in LaTeX format.")
    args = parser.parse_args()

    # Read the single CSV file
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file not found: {args.input_csv}")
        return
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading {args.input_csv}: {e}")
        return

    if df.empty:
        print(f"Input CSV file {args.input_csv} is empty.")
        return

    # Calculate global stats for models
    global_stats = compute_global_stats(df)
    # Calculate entity stats for judges
    entity_stats = compute_entity_stats(df, "_assessment")

    # Print global stats
    print("Global Model Stats:")
    print(tabulate(global_stats, headers="keys", tablefmt="pretty"))
    print("\nEntity (Judge) Stats:")
    print(tabulate(entity_stats, headers="keys", tablefmt="pretty"))

    if args.latex:
        latex_global = format_latex_table(global_stats, "Global Model Statistics", "tab:global-model-stats")
        latex_entity = format_latex_table(entity_stats, "Judge Statistics", "tab:judge-stats")
        print("\nLaTeX Global Model Stats:")
        print(latex_global)
        print("\nLaTeX Entity (Judge) Stats:")
        print(latex_entity)

if __name__ == "__main__":
    main()
