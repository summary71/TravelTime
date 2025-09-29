# performance_graph.py
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import colormaps as mcm  # 최신 API

REQUIRED_COLS = {"actual_values", "predictions", "reldiffs"}

def parse_pairs_json(pair_args):
    pairs = []
    for p in pair_args:
        try:
            obj = json.loads(p)
        except json.JSONDecodeError as e:
            raise ValueError(
                "--pair must be valid JSON like "
                '\'{"label":"My Model","file":"path.csv"}\'.\n'
                f"Got: {p}\nError: {e}"
            )
        label = obj.get("label") or obj.get("name") or obj.get("model")
        path  = obj.get("file")  or obj.get("path")
        if not label or not path:
            raise ValueError(f"--pair JSON must include 'label' and 'file' keys. Got: {p}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        pairs.append((str(label), path))  # 입력 순서 유지
    return pairs

def compute_metrics(df, rel_thresholds=(0.10, 0.20, 0.30)):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    y = df["actual_values"].to_numpy(dtype=float)
    yhat = df["predictions"].to_numpy(dtype=float)
    reldiffs = df["reldiffs"].to_numpy(dtype=float)

    if y.size == 0:
        # 빈 데이터 안전 처리
        nan = float("nan")
        within = {f"cdf_rel<={t}": nan for t in rel_thresholds}
        return {
            "MAE": nan, "RMSE": nan,
            "mean_reldiff": nan, "p10_reldiff": nan, "p50_reldiff": nan, "p90_reldiff": nan,
            **within
        }

    mae  = float(np.mean(np.abs(yhat - y)))
    rmse = float(sqrt(np.mean((yhat - y)**2)))
    mean_reldiff = float(np.mean(reldiffs))
    # reldiffs가 너무 짧으면 percentile 에러 방지
    if reldiffs.size >= 1:
        p10, p50, p90 = np.percentile(reldiffs, [10, 50, 90])
    else:
        p10 = p50 = p90 = float("nan")
    within = {f"cdf_rel<={t}": float(np.mean(reldiffs <= t)) for t in rel_thresholds}
    return {
        "MAE": mae, "RMSE": rmse,
        "mean_reldiff": mean_reldiff,
        "p10_reldiff": float(p10), "p50_reldiff": float(p50), "p90_reldiff": float(p90),
        **within
    }

def build_label_color_map(all_labels):
    """입력 라벨 순서에 맞춰 안정적인 색상 매핑 생성 (Matplotlib 3.7+ 호환)"""
    N = len(all_labels)
    if N <= 10:
        cmap = mcm.get_cmap('tab10')
        colors = list(cmap.colors)[:N]  # 정확히 10개 중 앞 N개
    elif N <= 20:
        cmap = mcm.get_cmap('tab20')
        colors = list(cmap.colors)[:N]  # 정확히 20개 중 앞 N개
    else:
        # 많은 모델: 연속형 팔레트에서 균등 샘플링 (시작색 반복 방지 위해 endpoint=False)
        cmap = mcm.get_cmap('hsv')
        colors = [cmap(x) for x in np.linspace(0, 1, N, endpoint=False)]
    return {lbl: colors[i] for i, lbl in enumerate(all_labels)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare prediction CSVs (JSON --pair entries) and plot CDF/MAE/RMSE.")
    ap.add_argument("--pair", action="append", required=True,
                    help='Repeatable. JSON with "label" and "file". '
                         'Example: --pair \'{"label":"DL (3 layers)","file":"results/dl.csv"}\'')
    ap.add_argument("--out_prefix", type=str, required=True,
                    help='Output filename prefix, e.g., "results/comparison_testdata". '
                         'Creates <prefix>.csv and three PDFs.')
    ap.add_argument("--title", type=str, default="", help="Optional plot title (e.g., test group).")
    ap.add_argument("--rel_thresholds", type=float, nargs="*", default=[0.10, 0.20, 0.30],
                    help="Relative-error thresholds for CDF percentages in the summary table.")
    args = ap.parse_args()

    pairs = parse_pairs_json(args.pair)  # 입력 순서 유지
    labels_in_order = [lbl for lbl, _ in pairs]

    rows = []
    cdf_series = []   # (label, reldiffs) in input order
    mae_map = {}      # label -> MAE
    rmse_map = {}     # label -> RMSE

    for label, path in pairs:
        df = pd.read_csv(path)
        metrics = compute_metrics(df, tuple(args.rel_thresholds))
        rows.append({"model_label": label, "file": os.path.basename(path), **metrics})
        cdf_series.append((label, df["reldiffs"].to_numpy(dtype=float)))
        mae_map[label]  = metrics["MAE"]
        rmse_map[label] = metrics["RMSE"]

    # 요약 표 (입력 순서 유지)
    summary = pd.DataFrame(rows)
    print("\n=== Comparison Summary (input order) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # 출력 디렉토리 준비
    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # CSV 저장
    out_csv = args.out_prefix + ".csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary to {out_csv}")

    # 공통 색상 매핑 (모든 플롯에 일관 적용)
    label2color = build_label_color_map(labels_in_order)

    # -------- Plot 1: Relative error CDF (입력 순서 및 색상 유지) --------
    plt.figure(figsize=(9, 6))
    for label, reldiffs in cdf_series:  # 입력 순서 그대로
        if reldiffs.size == 0:
            continue
        s = np.sort(reldiffs)
        c = np.arange(1, len(s) + 1) / len(s)
        plt.plot(s, c, label=label, color=label2color[label])
    plt.xlabel("Relative error")
    plt.ylabel("Cumulative probability")
    title = "CDF of Relative Error"
    if args.title:
        title += f" ({args.title})"
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    plt.tight_layout()
    cdf_path = args.out_prefix + "_relative_error_cdf.pdf"
    plt.savefig(cdf_path)
    print(f"Saved: {cdf_path}")
    plt.close()

    # -------- Plot 2: MAE (세로 막대, 입력 순서 & 색상 유지) --------
    labels_mae  = labels_in_order
    values_mae  = [mae_map[lbl] for lbl in labels_mae]
    colors_mae  = [label2color[lbl] for lbl in labels_mae]

    plt.figure(figsize=(max(9, 0.3 * len(labels_mae) + 3), 6))
    ax = plt.gca()
    x_pos = np.arange(len(labels_mae))
    ax.bar(x_pos, values_mae, color=colors_mae)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_mae, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("MAE")
    title = "MAE by Model"
    if args.title:
        title += f" ({args.title})"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(values_mae):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.4f}", va="bottom", ha="center", fontsize=9)
    plt.tight_layout()
    mae_path = args.out_prefix + "_mae.pdf"
    plt.savefig(mae_path)
    print(f"Saved: {mae_path}")
    plt.close()

    # -------- Plot 3: RMSE (세로 막대, 입력 순서 & 색상 유지) --------
    labels_rmse  = labels_in_order
    values_rmse  = [rmse_map[lbl] for lbl in labels_rmse]
    colors_rmse  = [label2color[lbl] for lbl in labels_rmse]

    plt.figure(figsize=(max(9, 0.3 * len(labels_rmse) + 3), 6))
    ax = plt.gca()
    x_pos = np.arange(len(labels_rmse))
    ax.bar(x_pos, values_rmse, color=colors_rmse)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_rmse, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("RMSE")
    title = "RMSE by Model"
    if args.title:
        title += f" ({args.title})"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(values_rmse):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.4f}", va="bottom", ha="center", fontsize=9)
    plt.tight_layout()
    rmse_path = args.out_prefix + "_rmse.pdf"
    plt.savefig(rmse_path)
    print(f"Saved: {rmse_path}")
    plt.close()
