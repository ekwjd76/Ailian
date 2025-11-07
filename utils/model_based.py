"""
model_based.py — 한국어 전용 로버타 기반 AI 탐지기
MIT License | 2025
"""

from transformers import pipeline
import torch

# ✅ 한국어 전용 모델 (문체 판별 성능 우수, MIT License)
MODEL_NAME = "klue/roberta-base"

DEVICE = 0 if torch.cuda.is_available() else -1

try:
    ai_detector = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=DEVICE,
        truncation=True
    )
    print(f"✅ Loaded Korean model: {MODEL_NAME} (device={DEVICE})")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    ai_detector = None


def model_ai_score(text: str) -> float:
    """
    한국어 문장을 입력받아 AI 작성 확률 추정 (0.0~1.0)
    """
    if not ai_detector:
        return 0.5

    try:
        snippet = text.strip()
        if len(snippet) > 1024:
            snippet = snippet[:1024]

        result = ai_detector(snippet)[0]
        label = result.get("label", "").upper()
        score = float(result.get("score", 0.5))

        # klue-roberta는 감성 분류용이므로 단독 사용 시 의미 약함
        # → 보정기와 함께 사용할 때 더 정확
        if "POSITIVE" in label or "AI" in label or "FAKE" in label:
            ai_score = score
        else:
            ai_score = 1.0 - score

        ai_score = max(0.0, min(1.0, ai_score))
        return ai_score

    except Exception as e:
        print(f"⚠️ model_ai_score error: {e}")
        return 0.5
