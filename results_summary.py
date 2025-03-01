#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import argparse
import glob
import os
import krippendorff
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, rankdata

parent_dir = Path(__file__).parent

# === Existing functions ====================================


def detect_judges(df):
    """Detect judges from assessment columns.

    For each column ending with "_assessment", split on '/' and use:
      - The last segment as the judge name.
      - The second-to-last segment (if it exists) as the provider.
    """
    judges = {}
    for col in df.columns:
        if col.endswith("_assessment"):
            parts = col[: -len("_assessment")].split("/")
            if len(parts) >= 2:
                judge_name = parts[-1]
                provider = parts[-2].lower()
            else:
                judge_name = parts[-1]
                provider = parts[-1].lower()
            judges[judge_name] = provider
    return list(judges.keys()), judges


def get_score_column(df, model):
    """Get score column for a model using the last segment of its name."""
    model_name = model.split("/")[-1]
    pat = re.compile(rf"^(?:.*/)?{re.escape(model_name)}_score\s*$", re.IGNORECASE)
    for col in df.columns:
        if pat.match(col):
            return col
    for col in df.columns:
        if model_name.lower() in col.lower() and (
            "score" in col.lower() or "assessment" in col.lower()
        ):
            return col
    return None


def mean_se_ci_str(vals, rd=3):
    """Calculate mean with standard error and 95% confidence interval.

    If only one value is present, SE is 0 and the CI equals the value.
    Returns a string of the form:
      "<mean> (<se>) [<lower>, <upper>]"
    """
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return "NaN"
    if len(arr) == 1:
        m = arr[0]
        return f"{m:.{rd}f} ({0:.{rd}f}) [{m:.{rd}f}, {m:.{rd}f}]"
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return f"{m:.{rd}f} ({se:.{rd}f}) [{m - 1.96 * se:.{rd}f}, {m + 1.96 * se:.{rd}f}]"


def compute_krippendorff_alpha(scores):
    """Compute Krippendorff's alpha for reliability using both interval and ordinal."""
    arr = np.array(scores)
    uniq = np.unique(arr[~np.isnan(arr)])
    if uniq.size <= 1:
        return np.nan, np.nan
    interval_alpha = krippendorff.alpha(
        reliability_data=arr, level_of_measurement="interval"
    )
    ordinal_alpha = krippendorff.alpha(
        reliability_data=arr, level_of_measurement="ordinal"
    )
    return interval_alpha, ordinal_alpha


def infer_provider(model):
    """Infer provider from a model name by returning the second-to-last segment if available."""
    parts = model.split("/")
    if len(parts) >= 2:
        return parts[-2].lower()
    return model.lower()


def get_related_judge_for_model(model, judges, judge_providers):
    """Return a related judge for a model.

    First, check if the model’s last segment exactly matches any judge name.
    If not, use infer_provider() and match against judge providers.
    """
    model_name = model.split("/")[-1].lower()
    for j in judges:
        if j.lower() == model_name:
            return j
    prov = infer_provider(model)
    for j, p in judge_providers.items():
        if p == prov:
            return j
    return None


def compute_global_judge_harshness(df, judges, judge_providers, rd=3):
    """
    For each judge j, consider each distinct model (from _answer columns) that is unrelated
    (i.e. infer_provider(model) != judge_providers[j]). For each such model, compute the
    average score given by judge j (over all completions for that model). Then, for judge j,
    compute the mean and SE/CI based on these model-level averages.
    """
    model_names = [
        col.rsplit("_answer", 1)[0] for col in df.columns if col.endswith("_answer")
    ]
    model_names = list(set(model_names))
    scores = {j: [] for j in judges}

    for model in model_names:
        model_prov = infer_provider(model)
        for j in judges:
            if model_prov == judge_providers[j]:
                continue
            answer_col = f"{model}_answer"
            model_rows = df[df[answer_col].notna()]
            if model_rows.empty:
                continue
            score_col = get_score_column(df, j)
            if score_col is None:
                continue
            vals = model_rows[score_col].dropna().astype(float).values
            if len(vals) > 0:
                model_avg = np.mean(vals)
                scores[j].append(model_avg)

    num, fmt = {}, {}
    for j in judges:
        if scores[j]:
            avg = np.mean(scores[j])
            n = len(scores[j])
            se = np.std(scores[j], ddof=1) / np.sqrt(n) if n > 1 else 0.0
            if n == 1:
                fmt[j] = f"{avg:.{rd}f} ({0:.{rd}f}) [{avg:.{rd}f}, {avg:.{rd}f}]"
            else:
                fmt[j] = (
                    f"{avg:.{rd}f} ({se:.{rd}f}) [{avg - 1.96 * se:.{rd}f}, {avg + 1.96 * se:.{rd}f}]"
                )
            num[j] = avg
        else:
            num[j], fmt[j] = np.nan, "n/a"
    return num, fmt


def calculate_judge_correlations(df, judges):
    """Calculate correlations between judges.

    For each judge pair, both Pearson and Spearman correlations are computed.
    Also calculates Krippendorff's alpha across judges (both interval and ordinal).
    """
    jscores = defaultdict(list)
    for _, row in df.iterrows():
        for j in judges:
            col = get_score_column(df, j)
            if col and pd.notna(row[col]):
                try:
                    jscores[j].append(float(row[col]))
                except:
                    continue
    counts = {j: len(jscores[j]) for j in judges}
    pair_results = []
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            j1, j2 = judges[i], judges[j]
            s1, s2 = np.array(jscores[j1]), np.array(jscores[j2])
            n_obs = min(len(s1), len(s2)) if len(s1) and len(s2) else 0
            if n_obs > 0:
                pearson_corr = pearsonr(s1[:n_obs], s2[:n_obs])[0]
                spearman_corr = spearmanr(s1[:n_obs], s2[:n_obs])[0]
            else:
                pearson_corr = np.nan
                spearman_corr = np.nan
            pair_results.append((f"{j1} vs {j2}", n_obs, pearson_corr, spearman_corr))

    # Calculate both types of Krippendorff's alpha
    jarrays = []
    if jscores:
        max_len = max(len(v) for v in jscores.values())
        for j in judges:
            arr = jscores[j][:]
            arr.extend([np.nan] * (max_len - len(arr)))
            jarrays.append(arr)
        interval_alpha, ordinal_alpha = compute_krippendorff_alpha(jarrays)
    else:
        interval_alpha, ordinal_alpha = np.nan, np.nan

    return pair_results, (interval_alpha, ordinal_alpha), counts


def process_combined_files(directory, calc_judge_harshness=False, rd=3):
    """Process combined results files with dynamic judge count.

    - 'completions' is the number of input files (from the glob) that contain the <model>_answer column.
    - 'n' is the number of rows in the combined DataFrame with a non-NaN answer for that model.

    Returns a tuple: (df_res, df, judges, judge_providers)
    """
    combined_files = [f for f in glob.glob(os.path.join(directory, "combined_*.csv"))]
    if not combined_files:
        print("No combined results files found.")
        return None

    models_set = set()
    completions_dict = {}
    for f in combined_files:
        try:
            df_file = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        for col in df_file.columns:
            if col.endswith("_answer"):
                model = col.rsplit("_answer", 1)[0]
                models_set.add(model)
                completions_dict[model] = completions_dict.get(model, 0) + 1
    models = list(models_set)

    dfs = []
    for f in combined_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)

    judges, judge_providers = detect_judges(df)

    if calc_judge_harshness:
        global_harshness_num, global_harshness_fmt = compute_global_judge_harshness(
            df, judges, judge_providers, rd
        )
    else:
        # Preset (hardcoded) values (excluding non-judge entries)
        preset = {
            "claude-3-5-sonnet-20241022": -0.411,
            "gemini-1.5-pro-002": 0.255,
            "gpt-4o-2024-08-06": 0.255,
        }
        global_harshness_num = {}
        global_harshness_fmt = {}
        for j in judges:
            if j in preset:
                global_harshness_num[j] = preset[j]
                global_harshness_fmt[j] = (
                    f"{preset[j]:.{rd}f} ({0:.{rd}f}) [{preset[j]:.{rd}f}, {preset[j]:.{rd}f}]"
                )
            else:
                global_harshness_num[j] = np.nan
                global_harshness_fmt[j] = "n/a"

    results = {}
    for model in models:
        answer_col = f"{model}_answer"
        assess_col = f"{model}_assessment"
        model_rows = df[df[answer_col].notna()]
        n_val = len(model_rows)
        completions = completions_dict.get(model, 0)
        is_judge = (
            assess_col in df.columns
            and (df[answer_col].notna() & df[assess_col].notna()).any()
        )
        related_judge = (
            None
            if is_judge
            else get_related_judge_for_model(model, judges, judge_providers)
        )
        role = "JUDGE" if is_judge else ("Related" if related_judge else "independent")

        raw_scores = []  # unadjusted row-level average
        adjusted_scores = []  # adjusted row-level average (with self/related score replaced)
        sp_values = []  # for display: responsible score minus judge harshness

        for _, row in model_rows.iterrows():
            rjs = {}
            for j in judges:
                score_col = get_score_column(df, j)
                if score_col and pd.notna(row[score_col]):
                    try:
                        val = float(row[score_col])
                        rjs[j] = val
                    except:
                        continue
            raw_val = np.mean(list(rjs.values())) if rjs else np.nan
            raw_scores.append(raw_val)

            # Build adjusted dictionary: replace self (or related) judge's score with its harshness.
            adjusted_rjs = dict(rjs)
            if role == "JUDGE":
                judge_name = model.split("/")[-1]
                if judge_name in adjusted_rjs:
                    adjusted_rjs[judge_name] = global_harshness_num.get(
                        judge_name, adjusted_rjs[judge_name]
                    )
            elif role == "Related":
                if related_judge is not None and related_judge in adjusted_rjs:
                    adjusted_rjs[related_judge] = global_harshness_num.get(
                        related_judge, adjusted_rjs[related_judge]
                    )
            adjusted_val = (
                np.mean(list(adjusted_rjs.values())) if adjusted_rjs else np.nan
            )
            adjusted_scores.append(adjusted_val)

            # Compute self-preference for display: responsible score minus harshness.
            if role == "JUDGE":
                score_col = get_score_column(df, model)
                if score_col and pd.notna(row[score_col]):
                    try:
                        responsible_score = float(row[score_col])
                    except:
                        responsible_score = np.nan
                else:
                    responsible_score = np.nan
                judge_name = model.split("/")[-1]
                harsh = global_harshness_num.get(judge_name, np.nan)
                sp_val = (
                    responsible_score - harsh
                    if (not np.isnan(responsible_score) and not np.isnan(harsh))
                    else np.nan
                )
            elif role == "Related":
                score_col = get_score_column(df, related_judge)
                if score_col and pd.notna(row[score_col]):
                    try:
                        responsible_score = float(row[score_col])
                    except:
                        responsible_score = np.nan
                else:
                    responsible_score = np.nan
                harsh = global_harshness_num.get(related_judge, np.nan)
                sp_val = (
                    responsible_score - harsh
                    if (not np.isnan(responsible_score) and not np.isnan(harsh))
                    else np.nan
                )
            else:
                sp_val = np.nan
            sp_values.append(sp_val)

        results[model] = {
            "Role": role,
            "Average Score": mean_se_ci_str(adjusted_scores, rd),  # Final Score
            "Unadjusted Average Score": mean_se_ci_str(raw_scores, rd),  # Raw Score
            "Self-Preference": mean_se_ci_str(sp_values, rd)
            if any(not np.isnan(x) for x in sp_values)
            else "n/a",
        }
        # Include additional info
        results[model]["# Trials"] = completions
        results[model]["Questions"] = n_val

        if role == "JUDGE":
            judge_name = model.split("/")[-1]
            results[model]["Related Judge Harshness"] = global_harshness_fmt.get(
                judge_name, "n/a"
            )
        elif role == "Related":
            results[model]["Related Judge Harshness"] = global_harshness_fmt.get(
                related_judge, "n/a"
            )
        else:
            results[model]["Related Judge Harshness"] = "n/a"

    # Reorder and rename columns for the summary table.
    ordered_cols = [
        "Average Score",
        "Role",
        "Unadjusted Average Score",
        "Self-Preference",
        "Related Judge Harshness",
        "# Trials",
        "Questions",
    ]
    df_res = pd.DataFrame(results).T
    df_res = df_res[ordered_cols]
    df_res.rename(
        columns={
            "Average Score": "Final Score",
            "Unadjusted Average Score": "Raw Score",
            "Related Judge Harshness": "Judge Harshness",
        },
        inplace=True,
    )

    # For sorting by Final Score, extract its numeric value.
    def extract_score(s):
        try:
            return float(s.split()[0])
        except:
            return np.nan

    df_res["FinalScoreNum"] = df_res["Final Score"].apply(extract_score)
    df_res = df_res.sort_values(by="FinalScoreNum", ascending=False).drop(
        columns=["FinalScoreNum"]
    )

    # Add ranking based on Final Score.
    df_res["FinalScoreNum"] = df_res["Final Score"].apply(extract_score)
    df_res["Rank"] = rankdata(-df_res["FinalScoreNum"], method="min")
    df_res = df_res.sort_values("Rank")
    df_res.drop(columns=["FinalScoreNum"], inplace=True)

    # Format the Questions column with commas.
    df_res["Questions"] = df_res["Questions"].apply(lambda x: f"{x:,}")

    return df_res, df, judges, judge_providers


# === Functions for LaTeX table formatting ===================


def format_final_score(s):
    """
    Reformat the adjusted average score string.
    Input is expected to be like:
      "0.022 (0.005) [0.012, 0.032]"
    and output:
      "\\textbf{0.022} (0.012, 0.032)"
    """
    if s in ["NaN", "n/a"]:
        return "--"
    m = re.match(r"([\d\.\-]+)\s+\([\d\.\-]+\)\s+\[([\d\.\-]+),\s*([\d\.\-]+)\]", s)
    if m:
        main, low, high = m.groups()
        return f"\\textbf{{{main}}} ({low}, {high})"
    return s


def format_simple_score(s):
    """
    Extract the main score (the first number) from strings of the form:
       "0.044 (0.011) [0.047, 0.089]"
    If not parseable, return "--".
    """
    if s in ["NaN", "n/a"]:
        return "--"
    try:
        return s.split()[0]
    except:
        return s


def format_latex_summary_table(df):
    """
    Format a LaTeX table for the summary results.
    Expected columns in df (in order):
      Final Score, Role, Raw Score, Self-Preference, Judge Harshness, # Trials, Questions.
    The output table will have two header rows.
    """
    latex_lines = [
        "\\begin{table}[ht]",
        "    \\setlength{\\tabcolsep}{6pt}",
        "    \\small",
        "    \\centering",
        "    \\begin{tabular}{lccccccc}",
        "        \\hline",
        "        \\textbf{Model} & \\textbf{Final Score} & \\textbf{Role} & \\textbf{Raw} & \\textbf{Self-} & \\textbf{Judge} & \\textbf{\\# Trials} & \\textbf{Questions} \\\\",
        "         & \\textbf{(95\\% CI)} &  & \\textbf{Score} & \\textbf{Preference} & \\textbf{Harshness} &  &  \\\\",
        "        \\hline",
    ]

    for model, row in df.iterrows():
        # Use only the last segment of the model name.
        model_name = model.split("/")[-1]

        # Format Role in small caps if it is 'JUDGE' or 'Related'.
        role = row["Role"]
        if role in ["JUDGE", "Related"]:
            role_fmt = f"\\textsc{{{role.lower()}}}"
        else:
            role_fmt = role

        final_score = format_final_score(row["Final Score"])
        raw_score = format_simple_score(row["Raw Score"])
        selfpref = format_simple_score(row["Self-Preference"])
        judge_harsh = format_simple_score(row["Judge Harshness"])
        trials = str(row["# Trials"])
        questions = row["Questions"]

        line = f"{model_name} & {final_score} & {role_fmt} & {raw_score} & {selfpref} & {judge_harsh} & {trials} & {questions} \\\\"
        latex_lines.append("        " + line)

    latex_lines.extend(
        [
            "        \\hline",
            "    \\end{tabular}",
            "    \\caption{\\textbf{Scores on the AHA benchmark.} Self-preference bias for compact models (e.g., Claude-3.5-Haiku) are based on the larger version of the model that we use as a judge (e.g., Claude-3.5-Sonnet). Final Score = Raw Score $-$ (Self-Preference $\\div$ 3).}",
            "    \\label{tab:judges}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex_lines)


# === Main ===================================================


def main():
    parser = argparse.ArgumentParser("Results Summary Analysis")
    parser.add_argument(
        "--input_directory",
        default=parent_dir / "results",
        help="Directory containing combined_*.csv files",
    )
    parser.add_argument(
        "--calculate_judge_harshness",
        action="store_true",
        help="If set, judge harshness is calculated from data (column header with asterisk).",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="If set, output the summary table in LaTeX format.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=3,
        help="Number of decimal places for rounding results (default: 3)",
    )
    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", True)

    result = process_combined_files(
        args.input_directory,
        calc_judge_harshness=args.calculate_judge_harshness,
        rd=args.round,
    )
    if result is not None:
        df_res, df, judges, judge_providers = result
        # For console output, change role to lower-case for judge/related.
        df_res["Role"] = df_res["Role"].apply(
            lambda x: x.lower() if x in ["JUDGE", "Related"] else x
        )

        # Compute judge correlations and overall Krippendorff's alphas
        pair_results, alphas, counts = calculate_judge_correlations(df, judges)
        interval_alpha, ordinal_alpha = alphas

        print("\nLLM-as-a-Judge scores with pooled runs (including rankings).\n")
        print(df_res.to_string())

        # Display judge correlations and overall Krippendorff's alphas
        print("\nJudge Correlations:")
        print(f"Overall Krippendorff's Alpha (interval): {interval_alpha:.3f}")
        print(f"Overall Krippendorff's Alpha (ordinal): {ordinal_alpha:.3f}")
        for pair, n_obs, pearson_corr, spearman_corr in pair_results:
            pearson_fmt = f"{pearson_corr:.3f}" if not np.isnan(pearson_corr) else "nan"
            spearman_fmt = (
                f"{spearman_corr:.3f}" if not np.isnan(spearman_corr) else "nan"
            )
            print(
                f"  {pair}: n = {n_obs}, Pearson = {pearson_fmt}, Spearman = {spearman_fmt}"
            )


if __name__ == "__main__":
    main()
