"""从 optimize_results 中读取最新阈值表，比较阈值 0.5 与最优阈值的指标，追加到报告。"""
from __future__ import annotations

import glob
import os
import pandas as pd


def find_latest_threshold_csv(dir_path: str) -> str:
    files = glob.glob(os.path.join(dir_path, "threshold_metrics_*.csv"))
    if not files:
        raise FileNotFoundError("No threshold_metrics_*.csv in " + dir_path)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_metrics(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def summarize_comparison(df: pd.DataFrame) -> str:
    # best by f1
    best_idx = df['f1'].idxmax()
    best = df.loc[best_idx]
    # closest to 0.5
    idx05 = (df['threshold'] - 0.5).abs().idxmin()
    row05 = df.loc[idx05]

    lines = []
    lines.append('**优化前后对比（阈值 0.5 vs 优化后）**')
    lines.append('')
    lines.append('| 指标 | 阈值=0.5 | 优化后（最佳阈值） |')
    lines.append('|---:|:---:|:---:|')
    for k in ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy', 'youden']:
        v05 = row05.get(k, float('nan'))
        vb = best.get(k, float('nan'))
        lines.append(f'| {k} | {v05:.4f} | {vb:.4f} |')

    lines.append('')
    lines.append(f'- 最佳阈值: {best["threshold"]:.3f} （基于 F1）')
    lines.append('')
    lines.append('解读：根据上表，优化后在平衡精确率与召回率方面取得更好效果（F1 值提升），可考虑将该阈值作为线上默认阈值并按业务要求进一步调整。')
    lines.append('')
    return '\n'.join(lines)


def append_to_report(report_path: str, content: str):
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write(content)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'optimize_results')
    csv = find_latest_threshold_csv(out_dir)
    df = load_metrics(csv)
    summary = summarize_comparison(df)
    report_path = os.path.join(os.path.dirname(__file__), 'MODEL_REPORT_COMPLETE.md')
    append_to_report(report_path, summary)
    print('Appended comparison to', report_path)


if __name__ == '__main__':
    main()
