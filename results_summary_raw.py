import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate

parent_dir = Path(__file__).parent


def compute_global_stats(df: pd.DataFrame):
    """
    Compute overall statistics from the combined dataframe.
    Returns:
      (number of model answer columns, number of judge assessment columns,
       mean score, score distribution, category distribution, total score count)
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    cat_cols = [c for c in df.columns if c.endswith("_category")]
    model_cols = [c for c in df.columns if c.endswith("_answer")]
    judge_cols = [c for c in df.columns if c.endswith("_assessment")]

    total_score = 0.0
    total_count = 0
    score_dist = {"-1": 0, "0": 0, "1": 0}
    cat_dist = {}

    for idx in df.index:
        for sc_col in score_cols:
            val = df.at[idx, sc_col]
            if pd.notna(val) and val in (-1, 0, 1):
                total_score += val
                total_count += 1
                score_dist[str(int(val))] += 1
        for c_col in cat_cols:
            c_val = str(df.at[idx, c_col]).strip()
            if c_val in {"A", "B", "C", "D", "E"}:
                cat_dist[c_val] = cat_dist.get(c_val, 0) + 1

    mean_val = total_score / total_count if total_count > 0 else 0.0
    ordered_cat = {k: cat_dist.get(k, 0) for k in sorted(["A", "B", "C", "D", "E"])}
    return (
        len(model_cols),
        len(judge_cols),
        mean_val,
        score_dist,
        ordered_cat,
        total_count,
    )


def compute_entity_stats(df: pd.DataFrame, suffix: str) -> dict:
    """
    For each entity (model or judge) determined by a column suffix,
    compute the number of questions answered, the score distribution, and category counts.
    """
    out = {}
    entity_cols = [c for c in df.columns if c.endswith(suffix)]
    is_judge = suffix == "_assessment"

    for col in entity_cols:
        name = col[: -len(suffix)]
        row_mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        relevant_indices = df.index[row_mask]

        q_count = len(relevant_indices)
        sc_count = 0
        sc_sum = 0.0
        sc_dist = {"-1": 0, "0": 0, "1": 0}
        c_dist = {}

        for rid in relevant_indices:
            if is_judge:
                score_col = f"{name}_score"
                cat_col = f"{name}_category"
                if score_col in df.columns:
                    val = df.at[rid, score_col]
                    if pd.notna(val) and val in (-1, 0, 1):
                        sc_sum += val
                        sc_count += 1
                        sc_dist[str(int(val))] += 1
                if cat_col in df.columns:
                    cat_val = str(df.at[rid, cat_col]).strip()
                    if cat_val in {"A", "B", "C", "D", "E"}:
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
                        if cat_val in {"A", "B", "C", "D", "E"}:
                            c_dist[cat_val] = c_dist.get(cat_val, 0) + 1

        ordered_c_dist = {
            k: c_dist.get(k, 0) for k in sorted(["A", "B", "C", "D", "E"])
        }
        out[name] = {
            "questions": q_count,
            "score_count": sc_count,
            "sum_score": sc_sum,
            "score_dist": sc_dist,
            "cat_dist": ordered_c_dist,
        }
    return out


def stats_to_row(name: str, stats: dict, format_type="console") -> list:
    """Convert stats dictionary to a row for the table with percentages"""
    total_scores = stats["score_count"]
    total_cats = sum(stats["cat_dist"].values())

    if format_type == "console":
        return [
            name,
            stats["questions"],
            stats["score_count"],
            f"{stats['sum_score'] / stats['score_count']:.3f}"
            if stats["score_count"] > 0
            else "0.000",
            f"{(stats['score_dist']['-1'] / total_scores * 100):.1f}"
            if total_scores > 0
            else "0.0",
            f"{(stats['score_dist']['0'] / total_scores * 100):.1f}"
            if total_scores > 0
            else "0.0",
            f"{(stats['score_dist']['1'] / total_scores * 100):.1f}"
            if total_scores > 0
            else "0.0",
            f"{(stats['cat_dist']['A'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['B'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['C'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['D'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['E'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
        ]
    else:  # latex
        avg_score = (
            stats["sum_score"] / stats["score_count"]
            if stats["score_count"] > 0
            else 0.000
        )
        return [
            name.replace("_", "\\_"),  # Escape underscores for LaTeX
            f"{stats['questions']:,}",
            f"{stats['score_count']:,}",
            f"{avg_score:.3f}",
            f"{(stats['score_dist']['-1'] / total_scores * 100):.1f}"
            if total_scores > 0
            else "0.0",
            f"{(stats['score_dist']['0'] / total_scores * 100):.1f}"
            if total_scores > 0
            else "0.0",
            f"{(stats['score_dist']['1'] / total_scores * 100):.1f}"
            if total_scores > 0
            else "0.0",
            f"{(stats['cat_dist']['A'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['B'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['C'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['D'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
            f"{(stats['cat_dist']['E'] / total_cats * 100):.1f}"
            if total_cats > 0
            else "0.0",
        ]


def format_latex_table(rows, caption, label):
    headers = [
        "Model",
        "Questions",
        "Scores",
        "Avg Score",
        "-1 (\\%)",
        "0 (\\%)",
        "1 (\\%)",
        "A (\\%)",
        "B (\\%)",
        "C (\\%)",
        "D (\\%)",
        "E (\\%)",
    ]

    latex = [
        "\\begin{table}[ht]",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\small",
        "    \\centering",
        "    \\begin{tabular}{l|cccccccccccc}",
        "        \\hline",
        "        \\textbf{" + "} & \\textbf{".join(headers) + "} \\\\",
        "        \\hline",
    ]

    # Add data rows
    for row in rows[:-1]:  # Exclude the total row for now
        latex.append("        " + " & ".join(str(x) for x in row) + " \\\\")

    # Add total row with hline
    latex.extend(
        [
            "        \\hline",
            "        " + " & ".join(str(x) for x in rows[-1]) + " \\\\",
            "        \\hline",
            "    \\end{tabular}",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary statistics from combined_*.csv files."
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        default=parent_dir / "results",
        help="Directory containing combined_*.csv files",
    )
    # New optional argument to enable LaTeX table output
    parser.add_argument(
        "--latex", action="store_true", help="Include LaTeX table output"
    )
    args = parser.parse_args()

    input_directory = Path(args.input_directory)
    if not input_directory.exists():
        print(f"Directory {input_directory} does not exist.")
        return

    csv_files = sorted(input_directory.glob("combined_*.csv"))
    if not csv_files:
        print(f"No combined_*.csv files found in {input_directory}.")
        return

    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not df_list:
        print("No CSV data could be loaded.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)

    # Compute all statistics
    nm, nj, mean_val, score_dist, cat_dist, s_count = compute_global_stats(combined_df)
    model_stats = compute_entity_stats(combined_df, "_answer")
    judge_stats = compute_entity_stats(combined_df, "_assessment")

    # Console headers
    console_headers = [
        "Name",
        "Questions",
        "Scores Count",
        "Average Score",
        "-1 (%)",
        "0 (%)",
        "1 (%)",
        "Cat A (%)",
        "Cat B (%)",
        "Cat C (%)",
        "Cat D (%)",
        "Cat E (%)",
    ]

    # Calculate total stats
    def calculate_total_stats(stats_dict):
        total_stats = {
            "questions": sum(s["questions"] for s in stats_dict.values()),
            "score_count": sum(s["score_count"] for s in stats_dict.values()),
            "sum_score": sum(s["sum_score"] for s in stats_dict.values()),
            "score_dist": {
                k: sum(s["score_dist"][k] for s in stats_dict.values())
                for k in ["-1", "0", "1"]
            },
            "cat_dist": {
                k: sum(s["cat_dist"][k] for s in stats_dict.values())
                for k in ["A", "B", "C", "D", "E"]
            },
        }
        return total_stats

    # Prepare console tables
    model_rows_console = [
        stats_to_row(name, stats, "console")
        for name, stats in sorted(model_stats.items())
    ]
    model_total = calculate_total_stats(model_stats)
    model_rows_console.append(stats_to_row("Total", model_total, "console"))

    judge_rows_console = [
        stats_to_row(name, stats, "console")
        for name, stats in sorted(judge_stats.items())
    ]
    judge_total = calculate_total_stats(judge_stats)
    judge_rows_console.append(stats_to_row("Total", judge_total, "console"))

    # Print console tables
    print("\nResults by Model:")
    print(
        tabulate(
            model_rows_console, headers=console_headers, tablefmt="pipe", floatfmt=".1f"
        )
    )

    print("\nResults by Judge:")
    print(
        tabulate(
            judge_rows_console, headers=console_headers, tablefmt="pipe", floatfmt=".1f"
        )
    )

    # Print LaTeX tables only if --latex is specified
    if args.latex:
        model_rows_latex = [
            stats_to_row(name, stats, "latex")
            for name, stats in sorted(model_stats.items())
        ]
        model_rows_latex.append(stats_to_row("Total", model_total, "latex"))

        judge_rows_latex = [
            stats_to_row(name, stats, "latex")
            for name, stats in sorted(judge_stats.items())
        ]
        judge_rows_latex.append(stats_to_row("Total", judge_total, "latex"))

        print("\nLaTeX table for Models:")
        print(
            format_latex_table(
                model_rows_latex, "Model Performance Statistics", "tab:model-results"
            )
        )

        print("\nLaTeX table for Judges:")
        print(
            format_latex_table(
                judge_rows_latex, "Judge Performance Statistics", "tab:judge-results"
            )
        )


if __name__ == "__main__":
    main()
