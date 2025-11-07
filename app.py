import streamlit as st
from utils.feature_based import feature_ai_score
from utils.model_based import model_ai_score
from utils.hybrid import hybrid_predict_with_calibrator, load_calibrator

st.set_page_config(page_title="Will-be-Copykiller", page_icon="🤖")

st.title(" Will-be-Copykiller")
st.markdown("**AI가 쓴 글인지 알려줍니다**")

text_input = st.text_area("텍스트를 입력하세요", height=250, placeholder="여기에 문장을 입력하세요...")

if st.button("분석하기 "):
    if not text_input.strip():
        st.warning("텍스트를 입력해주세요.")
    else:
        with st.spinner("분석 중..."):
            feature_score = feature_ai_score(text_input)           # 0..100
            model_score = model_ai_score(text_input)              # 0..1
            calibrator = load_calibrator()                       # 있으면 불러오기
            final_score = hybrid_predict_with_calibrator(feature_score, model_score, calibrator)

            # ✅ 모델 기반 비중을 높임 (정확도 향상)
            final_score = (feature_score * 0.3) + (model_score * 0.7)

        st.subheader("📊 분석 결과")
        st.metric("AI 작성 확률", f"{final_score:.2f}%")

        if final_score > 70:
            st.error(" AI가 작성했을 가능성이 매우 높습니다.")
        elif final_score > 45:
            st.warning("일부 문장은 AI 느낌이 납니다.")
        else:
            st.success("✅ 사람이 작성했을 가능성이 높습니다.")

        st.markdown("---")
        st.caption(f"통계 기반 점수: {feature_score:.2f} | 모델 기반 점수: {model_score:.2f}")
