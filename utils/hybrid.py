"""
utils/hybrid.py

- hybrid_predict(feature_score, model_score, text_features=None): 즉시 사용 가능한 예측기
- train_calibrator(csv_path, out_path): 로컬 라벨 데이터로 로지스틱 회귀 보정 모델 학습 (선택 기능)

CSV 포맷(예시):
text,label
"이 문장은 사람이 쓴 예시입니다.",human
"GPT가 생성한 예시입니다.",ai
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 기본 가중치(모델 신뢰도를 높게 둠)
DEFAULT_WEIGHTS = {"feature": 0.3, "model": 0.7}

# 파일에 저장할 기본 경로
CALIBRATOR_PATH = "utils/hybrid_calibrator.pkl"


def hybrid_predict(feature_score: float, model_score: float) -> float:
    """
    빠른 하이브리드 결합: feature_score (0~100), model_score (0~1)
    반환값: 최종 AI 확률 (0~100)
    기본: final = 100 * (w_f * feature/100 + w_m * model_score)
    """
    w_f = DEFAULT_WEIGHTS["feature"]
    w_m = DEFAULT_WEIGHTS["model"]

    final = (w_f * (feature_score / 100.0) + w_m * model_score)
    return max(0.0, min(1.0, final)) * 100.0


def _prepare_features(df: pd.DataFrame):
    """
    df must have columns: 'text' and 'label' (label in {'ai','human'}).
    Returns X (n x 2): [feature_score_norm, model_score], y (0/1)
    """
    from utils.feature_based import feature_ai_score
    from utils.model_based import model_ai_score

    X_list = []
    y_list = []
    for _, row in df.iterrows():
        text = str(row["text"])
        label = row["label"].strip().lower()
        feature = feature_ai_score(text) / 100.0  # 0~1
        model = model_ai_score(text)  # 0~1
        X_list.append([feature, model])
        y_list.append(1 if label in ("ai", "generated", "bot") else 0)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def train_calibrator(csv_path: str, out_path: Optional[str] = None, test_size=0.2, random_state=42):
    """
    Train a simple logistic regression calibrator from labeled CSV.
    CSV must have 'text' and 'label' columns. label values: 'ai' or 'human' (case-insensitive)
    Saves the fitted sklearn model to out_path (default: utils/hybrid_calibrator.pkl).
    Prints out evaluation metrics.
    """
    if out_path is None:
        out_path = CALIBRATOR_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    if csv_path.endswith(".xlsx"):
        df = pd.read_excel(csv_path, engine="openpyxl")
    else:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    X, y = _prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("=== Calibrator evaluation ===")
    print(classification_report(y_test, y_pred, digits=4))

    # save model
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"✅ Calibrator saved to {out_path}")
    return clf


def load_calibrator(path: Optional[str] = None):
    if path is None:
        path = CALIBRATOR_PATH
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        clf = pickle.load(f)
    return clf


def hybrid_predict_with_calibrator(feature_score: float, model_score: float, calibrator=None):
    """
    If calibrator is provided (sklearn model), use it to compute probability.
    Else fallback to hybrid_predict.
    feature_score: 0~100, model_score: 0~1
    Returns final AI probability (0~100)
    """
    if calibrator is None:
        calibrator = load_calibrator()
    if calibrator is None:
        return hybrid_predict(feature_score, model_score)

    X = np.array([[feature_score / 100.0, model_score]])
    prob = calibrator.predict_proba(X)[0][1]
    return float(prob * 100.0)
