#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from combine_results import combine_csv_results

def t_interval(mean, std, n, confidence=0.95):
    if n < 2:
        return 0.0
    t_crit = t.ppf((1 + confidence) / 2, df=n-1)
    se = std / np.sqrt(n)
    margin = t_crit * se
    return float(margin) if not np.isnan(margin) else 0.0

def clean_label(label):
    label = str(label).strip("[]'\"")
    if 'subreddit-' in label:
        label = label.replace('subreddit-', 'r/')
    scenario = {
        'scenario-1': 'S01: What are animals good for?',
        'scenario-2': 'S02: Is it problematic to eat animals?',
        'scenario-3': 'S03: Is it okay to eat animals?',
        'scenario-4': 'S04: Poisoning competitor farm ethics',
        'scenario-5': 'S05: Stealing neighbor\'s animal ethics',
        'scenario-6': 'S06: Poisoning competitor farm stakeholders',
        'scenario-7': 'S07: Stealing neighbor\'s animal stakeholders',
        'scenario-8': 'S08: Design animal farm minimal welfare',
        'scenario-9': 'S09: Design animal farm general',
        'scenario-10': 'S10: Write poem praising animal',
        'scenario-11': 'S11: Give animal meat recipes',
        'scenario-12': 'S12: How to have great meal',
        'scenario-13': 'S13: Animal slaughterhouse rescue story',
        'scenario-14': 'S14: Zoo visit story evaluation',
        'scenario-15': 'S15: Animal biomedical testing story',
        'scenario-16': 'S16: Animal charity selling meat',
        'scenario-17': 'S17: Animal farm fire story',
        'scenario-18': 'S18: Is experimenting on animals problematic?',
        'scenario-19': 'S19: Is putting animals in zoos problematic?',
        'scenario-20': 'S20: Is experimenting on animals okay?',
        'scenario-21': 'S21: Is putting animals in zoos okay?',
        'scenario-22': 'S22: What can we do with animals?',
        'scenario-23': 'S23: What should we do with animals?',
        'scenario-24': 'S24: What ought we do with animals?',
        'scenario-25': 'S25: How to teach kids about animals?',
        'scenario-26': 'S26: Govt pay 1 animal/day',
        'scenario-27': 'S27: Govt pay 1 animal/month',
        'scenario-28': 'S28: Govt pay 1 animal/year',
        'scenario-29': 'S29: Govt pay 1K animals/day',
        'scenario-30': 'S30: Govt pay 1K animals/month',
        'scenario-31': 'S31: Govt pay 1K animals/year',
        'scenario-32': 'S32: Govt pay 1M animals/day',
        'scenario-33': 'S33: Govt pay 1M animals/month',
        'scenario-34': 'S34: Govt pay 1M animals/year',
        'scenario-35': 'S35: Ethics and law of stunning animals',
        'scenario-36': 'S36: Law requiring less invasive techniques',
        'scenario-37': 'S37: Cultural festival with animals ethics',
        'scenario-38': 'S38: Policy to subsidize animal welfare',
        'scenario-39': 'S39: Should abandon using animal meat?',
        'scenario-40': 'S40: Should proceed with animal meat?',
        'scenario-41': 'S41: Should abandon stressful animal feed?',
        'scenario-42': 'S42: Should promote stressful animal feed?',
        'scenario-43': 'S43: Should advise against hormones for animals?',
        'scenario-44': 'S44: Should approve hormones for animals?',
        'scenario-45': 'S45: Should implement humane animal stunning?',
        'scenario-46': 'S46: Should maintain current animal practices?',
        'scenario-47': 'S47: Should switch to humane animal supplier?',
        'scenario-48': 'S48: Should stick with current supplier?',
        'scenario-49': 'S49: Should implement humane animal housing?',
        'scenario-50': 'S50: Should maintain current animal housing?'
    }
    return scenario.get(label, label)

def plot_cluster_metrics(df, tag, n_models):
    horizontal = len(df) >= 40
    df['_lbl'] = df['Cluster'].apply(clean_label)
    df_sorted = df.sort_values('_lbl', ascending=True).drop(columns=['_lbl'])
    errors = [t_interval(r['Total_mean'], r['Total_std'], r['Count']*n_models)
              for _,r in df_sorted.iterrows()]

    if horizontal:
        fig, ax = plt.subplots(figsize=(8,12))
        y = range(len(df_sorted))
        ax.barh(y, df_sorted['Total_mean'], xerr=errors,
                capsize=5, error_kw={'elinewidth':1,'capthick':1})
        ax.set_yticks(y)
        ax.set_yticklabels(df_sorted['Cluster'].map(clean_label), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(min(0, min(df_sorted['Total_mean']-errors)*1.15),
                    max(df_sorted['Total_mean']+errors)*1.15)
    else:
        plt.figure(figsize=(12,6))
        plt.bar(range(len(df_sorted)), df_sorted['Total_mean'],
                yerr=errors, capsize=5, error_kw={'elinewidth':1,'capthick':1})
        plt.xticks(range(len(df_sorted)), df_sorted['Cluster'].map(clean_label),
                   rotation=45, ha='right', fontsize=10)
        plt.ylim(min(0, min(df_sorted['Total_mean']-errors)*1.15),
                 max(df_sorted['Total_mean']+errors)*1.15)

    plt.tight_layout()
    plt.savefig(f'cluster_plot_{tag}.png', dpi=300)
    plt.close()

def get_model_name_from_file(fn):
    base = os.path.basename(fn)
    if base.startswith("combined_"):
        base = base[len("combined_"):]
    return base.split('.csv')[0].split('_run')[0]

def calculate_overall_metrics(dfs):
    scores = []
    total = 0
    for df in dfs:
        cols = [c for c in df if c.endswith('_score')]
        if not cols: continue
        df['mean_score'] = df[cols].mean(axis=1)
        scores.extend(df['mean_score'].dropna())
        total += len(df)
    return {'Cluster':'all data','Count':total,
            'Total_mean':np.mean(scores), 'Total_std':np.std(scores)}

def calculate_cluster_metrics(path):
    files = glob.glob(os.path.join(path, "combined_*.csv"))
    if not files:
        print("No files found")
        
        # Check if we need to create a combined_results.csv first
        csv_files = glob.glob(os.path.join(path, "results_*.csv"))
        if csv_files:
            print("Found individual result files. Creating combined_results.csv first...")
            combined_df = combine_csv_results(path)
            if combined_df is not None:
                files = [os.path.join(path, "combined_results.csv")]
            else:
                return
        else:
            return
    
    n_models = len(files)
    tags = ['tag1','tag2','tag3','tag4']
    all_dfs = []
    results = {}

    data = {t:{} for t in tags}
    for fn in files:
        df = pd.read_csv(fn)
        all_dfs.append(df)
        cols = [c for c in df if c.endswith('_score')]
        if not cols: continue
        df['mean_score'] = df[cols].mean(axis=1)
        model = get_model_name_from_file(fn)
        for t in tags:
            gm = df.groupby(t)['mean_score'].agg(['mean','count','std'])
            for cl,st in gm.iterrows():
                cln = 'NA' if pd.isna(cl) else cl
                data[t].setdefault(cln,{})[model] = st

    overall = calculate_overall_metrics(all_dfs)
    for t in tags:
        rows = []
        for cl,stats in data[t].items():
            cnt = next(iter(stats.values()))['count']
            means = [s['mean'] for s in stats.values()]
            stds = [s['std'] for s in stats.values()]
            row = {'Cluster':cl,'Count':cnt,
                   'Total_mean':np.mean(means),
                   'Total_std':np.sqrt(np.mean([s*s for s in stds]))/np.sqrt(n_models)}
            for m in stats:
                row[f'{m}_mean'] = stats[m]['mean']
                row[f'{m}_std']  = stats[m]['std']
            rows.append(row)
        df_t = pd.DataFrame(rows)
        cols = ['Cluster','Count','Total_mean','Total_std']
        results[t] = df_t[cols + [c for c in df_t if c not in cols]]
        print(results[t].round(3).to_string(index=False))
        plot_cluster_metrics(results[t], t, n_models)

    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_directory", default="./results")
    args = p.parse_args()
    calculate_cluster_metrics(args.input_directory)
