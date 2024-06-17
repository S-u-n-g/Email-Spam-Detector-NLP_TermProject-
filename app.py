import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Streamlit 웹 애플리케이션 제목
st.title("Email Spam Detection")

# 사용자 입력을 위한 텍스트 박스
user_input = st.text_area("Enter text to analyze:")

# 모델과 토크나이저 로드
model_name = "spam_detection_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 파이프라인 생성
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 분석 버튼
if st.button("Analyze"):
    if user_input:
        # 스팸 탐지 수행
        result = nlp(user_input)
        label = result[0]['label']
        score = result[0]['score']

        # 결과 출력
        st.write(f"Input Text: {user_input}")
        if label == "LABEL_1":
            st.markdown(f"<h3>Prediction: <span style='color:red;'>Spam</span> with a probability of {score:.2f}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3>Prediction: <span style='color:green;'>Not Spam</span> with a probability of {score:.2f}</h3>", unsafe_allow_html=True)
    else:
        st.write("Please enter text to analyze.")
