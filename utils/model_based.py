"""
model_based.py — 한국어 전용 로버타 기반 AI 탐지기 (보정기 미적용)
MIT License | 2025
"""

from transformers import pipeline
import torch

# -------------------------------
# 모델 설정
# -------------------------------
MODEL_NAME = "MLP-KTLim/roberta-base-korean-ai-detector"
DEVICE = 0 if torch.cuda.is_available() else -1

# -------------------------------
# 모델 로드
# -------------------------------
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

# -------------------------------
# 모델 기반 AI 점수
# -------------------------------
def model_ai_score(text: str) -> float:
    """
    KLUE 모델을 사용한 한국어 AI 작성 확률 추정 (0~1)
    보정기 없이 단독 점수만 사용
    """
    if not ai_detector:
        return 0.5  # 모델 없으면 중립값

    try:
        snippet = text.strip()
        if len(snippet) > 1024:
            snippet = snippet[:1024]

        result = ai_detector(snippet)[0]
        label = result.get("label", "").upper()
        score = float(result.get("score", 0.5))

        # 단순 POSITIVE/NEGATIVE 해석
        if "POSITIVE" in label:
            ai_score = score
        elif "NEGATIVE" in label:
            ai_score = 1.0 - score
        else:
            ai_score = 0.5

        ai_score = max(0.0, min(1.0, ai_score))
        return ai_score

    except Exception as e:
        print(f"⚠️ model_ai_score error: {e}")
        return 0.5

# -------------------------------
# 테스트
# -------------------------------
if __name__ == "__main__":
    test_texts = [
        "이 문장은 AI가 작성했을 가능성이 있는 예시 문장입니다.",
        "오늘 날씨가 맑고 기분이 좋다.",
        "챗GPT를 활용한 자동 보고서 작성 예제입니다."
    ]
    
    for t in test_texts:
        print("문장:", t)
        print("Model score:", model_ai_score(t))
        print("-"*40)
