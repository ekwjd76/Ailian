# feature_based.py
import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# punkt / punkt_tab 자동 다운로드
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# AI가 자주 사용하는 경향이 있는 한국어 접속사 및 관용구 리스트
# 이 리스트는 실제 데이터를 통해 계속 보강해야 합니다.
AI_FAVORITE_TOKENS = [
    "따라서", "결론적으로", "마지막으로", "또한", "한편", 
    "그러므로", "이를", "바탕으로", "이러한", "점에서", 
    "수", "있습니다", "할", "수", "있다", "것으로", "보인다",
    "매우", "다양한", "효과적인", "본질적으로"
]

def feature_ai_score(text: str) -> float:
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    if not words or not sentences:
        return 50.0

    num_words = len(words)
    num_sentences = len(sentences)

    # 1. 평균 문장 길이 (Average Sentence Length - ASL)
    # AI는 문장이 길어지는 경향이 있음. 기준 15단어
    avg_sentence_length = sum(len(s.split()) for s in sentences) / num_sentences

    # 2. 어휘 다양성 (Lexical Diversity - LD)
    # AI는 반복적인 단어를 사용하는 경향이 있음 (LD가 낮음)
    lexical_diversity = len(set(words)) / num_words

    # 3. AI 선호 토큰 빈도 (AI Favorite Token Frequency - AIFT)
    # AI_FAVORITE_TOKENS가 전체 단어에서 차지하는 비율
    ai_token_count = sum(1 for word in words if word in AI_FAVORITE_TOKENS)
    ai_token_frequency = (ai_token_count / num_words) * 100 # 100단어당 빈도로 계산

    # --- 스코어 계산 (가중치 조정 필요) ---
    
    # 1. ASL 점수: 15단어 초과 시 가산점 (20점 만점)
    asl_score = max(avg_sentence_length - 15, 0) * 1.3 
    asl_score = min(asl_score, 20)
    
    # 2. LD 점수: 다양성 낮을수록 (1-LD) 점수 높음 (50점 만점)
    ld_score = (1 - lexical_diversity) * 50
    
    # 3. AIFT 점수: 빈도가 높을수록 점수 높음 (30점 만점)
    aift_score = ai_token_frequency * 4
    aift_score = min(aift_score, 30)

    # 총 점수 합산 및 0-100% 클리핑
    # 총점: LD(50) + AIFT(30) + ASL(20) = 100
    score = ld_score + aift_score + asl_score
    return min(max(score, 0), 100)