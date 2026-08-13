from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FEATURES, KOREAN_LABELS, LABELS


PLOT_FEATURE_LABELS = {
    "Age": "Age",
    "SystolicBP": "Systolic BP",
    "DiastolicBP": "Diastolic BP",
    "BS": "Blood sugar",
    "BodyTemp": "Body temperature",
    "HeartRate": "Heart rate",
}


def save_figures(
    clean_frame: pd.DataFrame,
    comparison: pd.DataFrame,
    best_model_name: str,
    confusion_matrix_values: list[list[int]],
    feature_importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    class_counts = clean_frame["RiskLevel"].astype("string").value_counts().reindex(LABELS)
    plt.figure(figsize=(7, 4.5))
    sns.barplot(x=LABELS, y=class_counts.values)
    plt.xlabel("Risk level")
    plt.ylabel("Count")
    plt.title("Risk-level distribution after preprocessing")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    correlation = clean_frame[FEATURES].corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature correlations")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 5.2))
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Holdout confusion matrix: {best_model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "best_model_confusion_matrix.png", dpi=180)
    plt.close()

    plot_importance = feature_importance.sort_values("importance_mean", ascending=True)
    plt.figure(figsize=(7, 4.8))
    plt.barh(
        [PLOT_FEATURE_LABELS[feature] for feature in plot_importance["feature"]],
        plot_importance["importance_mean"],
        xerr=plot_importance["importance_std"],
    )
    plt.xlabel("Decrease in holdout macro F1")
    plt.title("Permutation importance (global holdout result)")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_feature_importance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    model_labels = comparison["model"].str.replace("_", " ")
    plt.bar(model_labels, comparison["cv_macro_f1_mean"])
    plt.errorbar(
        model_labels,
        comparison["cv_macro_f1_mean"],
        yerr=comparison["cv_macro_f1_std"],
        fmt="none",
        color="black",
        capsize=4,
    )
    plt.ylim(0, 1)
    plt.ylabel("Cross-validated macro F1")
    plt.title("Candidate-model comparison")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


def write_results_markdown(
    output_path: Path,
    audit: dict,
    comparison: pd.DataFrame,
    best_model_name: str,
    detailed_metrics: dict,
) -> None:
    best_row = comparison.loc[comparison["model"] == best_model_name].iloc[0]
    class_report = detailed_metrics[best_model_name]
    comparison_header = (
        "| 모델 | CV macro F1 | CV 표준편차 | 홀드아웃 정확도 | "
        "홀드아웃 balanced accuracy | 홀드아웃 macro F1 |"
    )
    comparison_rule = "|---|---:|---:|---:|---:|---:|"
    comparison_lines = [comparison_header, comparison_rule]
    for _, row in comparison.iterrows():
        comparison_lines.append(
            f"| {row['model']} | {row['cv_macro_f1_mean']:.4f} | "
            f"{row['cv_macro_f1_std']:.4f} | {row['holdout_accuracy']:.4f} | "
            f"{row['holdout_balanced_accuracy']:.4f} | {row['holdout_macro_f1']:.4f} |"
        )

    lines = [
        "# 실행 결과",
        "",
        "이 문서는 `python run_pipeline.py` 실행 시 실제 결과로 다시 생성됩니다.",
        "",
        "## 데이터 처리 결과",
        "",
        f"- 공식 CSV 원본: {audit['raw_rows']:,}행",
        f"- 원본의 완전 동일 중복: {audit['raw_exact_duplicate_rows']:,}행",
        f"- 중복 제거 및 심박수 7 bpm 보정 후: {audit['clean_rows']:,}행",
        f"- 결측치: {audit['clean_missing_values']}개",
        "",
        "## 모델 비교",
        "",
        *comparison_lines,
        "",
        "## 최종 모델",
        "",
        f"- 선택 모델: `{best_model_name}`",
        "- 선택 기준: 학습 세트 내부 5겹 층화 교차검증 macro F1",
        f"- 교차검증 macro F1: {best_row['cv_macro_f1_mean']:.4f} ± {best_row['cv_macro_f1_std']:.4f}",
        f"- 독립 홀드아웃 정확도: {best_row['holdout_accuracy']:.4f}",
        f"- 독립 홀드아웃 balanced accuracy: {best_row['holdout_balanced_accuracy']:.4f}",
        f"- 독립 홀드아웃 macro F1: {best_row['holdout_macro_f1']:.4f}",
        "",
        "### 위험 단계별 홀드아웃 성능",
        "",
        "| 단계 | 정밀도 | 재현율 | F1 | 표본 수 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        values = class_report[label]
        lines.append(
            f"| {KOREAN_LABELS[label]} | {values['precision']:.4f} | "
            f"{values['recall']:.4f} | {values['f1-score']:.4f} | {int(values['support'])} |"
        )
    lines.extend(
        [
            "",
            "> 이 수치는 452행의 정제 데이터에서 한 번 분리한 20% 내부 검증 세트의 결과입니다. "
            "외부 병원 데이터로 검증한 임상 성능이 아니며 진단 목적으로 사용할 수 없습니다.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
