from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    FEATURE_LABELS,
    FEATURES,
    KOREAN_LABELS,
    METADATA_PATH,
    MODEL_PATH,
    UCI_DATASET_PAGE,
)


st.set_page_config(
    page_title="임산부 건강 위험도 분류",
    page_icon="🤰",
    layout="centered",
)


@st.cache_resource
def load_artifacts(model_path: Path, metadata_path: Path):
    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    return model, metadata


st.title("임산부 건강 위험도 분류 데모")
st.caption("6개 건강지표를 입력해 학습된 머신러닝 모델의 분류 결과를 확인합니다.")
st.warning(
    "교육·연구용 데모입니다. 의료기기가 아니며 진단, 응급 판단 또는 치료 결정에 "
    "사용할 수 없습니다. 건강 이상이나 우려가 있으면 의료진에게 상담하세요."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("학습 결과 파일이 없습니다. 터미널에서 `python run_pipeline.py`를 실행하세요.")
    st.stop()

model, metadata = load_artifacts(MODEL_PATH, METADATA_PATH)
ranges = metadata["feature_ranges_observed_in_clean_data"]

integer_features = {"Age", "SystolicBP", "DiastolicBP", "HeartRate"}
input_values: dict[str, float] = {}

with st.form("risk_input_form"):
    st.subheader("건강지표 입력")
    st.caption(
        "입력 가능 범위는 의학적 정상 범위가 아니라, 모델이 학습한 정제 데이터의 관측 범위입니다."
    )
    left, right = st.columns(2)
    for index, feature in enumerate(FEATURES):
        bounds = ranges[feature]
        target_column = left if index % 2 == 0 else right
        if feature in integer_features:
            value = target_column.number_input(
                FEATURE_LABELS[feature],
                min_value=int(bounds["min"]),
                max_value=int(bounds["max"]),
                value=int(round(bounds["median"])),
                step=1,
                help=f"학습 데이터 관측 범위: {bounds['min']:.0f}~{bounds['max']:.0f}",
            )
        else:
            value = target_column.number_input(
                FEATURE_LABELS[feature],
                min_value=float(bounds["min"]),
                max_value=float(bounds["max"]),
                value=float(bounds["median"]),
                step=0.1,
                format="%.1f",
                help=f"학습 데이터 관측 범위: {bounds['min']:.1f}~{bounds['max']:.1f}",
            )
        input_values[feature] = float(value)
    submitted = st.form_submit_button("위험도 분류", type="primary", width="stretch")

if submitted:
    input_frame = pd.DataFrame([input_values], columns=FEATURES)
    prediction = str(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]
    probability_map = dict(zip(model.classes_, probabilities))

    if prediction == "high risk":
        st.error(f"모델 분류 결과: **{KOREAN_LABELS[prediction]}**")
    elif prediction == "mid risk":
        st.warning(f"모델 분류 결과: **{KOREAN_LABELS[prediction]}**")
    else:
        st.success(f"모델 분류 결과: **{KOREAN_LABELS[prediction]}**")

    probability_frame = pd.DataFrame(
        {
            "위험 단계": [KOREAN_LABELS[label] for label in metadata["labels"]],
            "모델 점수": [probability_map[label] for label in metadata["labels"]],
        }
    ).set_index("위험 단계")
    st.bar_chart(probability_frame, y="모델 점수", horizontal=True)
    st.caption(
        "표시된 값은 이 모델이 계산한 분류 점수이며, 임상적으로 보정된 실제 질병 확률이 아닙니다."
    )

with st.expander("모델과 데이터 정보"):
    metrics = metadata["selected_model_metrics"]
    st.write(f"선택 모델: `{metadata['selected_model']}`")
    st.write(f"학습 내부 교차검증 macro F1: {metrics['cv_macro_f1_mean']:.3f}")
    st.write(f"독립 홀드아웃 macro F1: {metrics['holdout_macro_f1']:.3f}")
    st.write(
        f"원본 {metadata['preprocessing']['raw_rows']:,}행 → "
        f"정제 {metadata['preprocessing']['clean_rows']:,}행"
    )
    st.markdown(f"데이터: [UCI Maternal Health Risk]({UCI_DATASET_PAGE})")

with st.expander("최종 모델의 전체 변수 중요도"):
    importance = pd.DataFrame(metadata["permutation_feature_importance"])
    importance["변수"] = importance["feature"].map(FEATURE_LABELS)
    st.dataframe(
        importance[["변수", "importance_mean", "importance_std"]].rename(
            columns={
                "importance_mean": "평균 중요도",
                "importance_std": "표준편차",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "홀드아웃 데이터에서 변수를 섞었을 때 macro F1이 얼마나 감소했는지 나타낸 "
        "전체 표본 기준 순열 중요도입니다. 개인별 원인이나 인과관계를 뜻하지 않습니다."
    )

st.divider()
st.caption("입력값은 이 앱에서 별도로 저장하지 않습니다.")
