import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import evaluate
import torch

# 디바이스 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 데이터셋 로드
file_path = './Balanced_SMS_Spam_Dataset.csv'
df = pd.read_csv(file_path)

# 데이터셋 확인
print(df.head())

from sklearn.utils import resample

# 스팸과 정상 메일의 비율 확인
spam_count = df[df['label'] == 1].shape[0]
non_spam_count = df[df['label'] == 0].shape[0]
print(f"스팸: {spam_count}, 정상: {non_spam_count}")

# 데이터셋을 균형 있게 만들기
spam_df = df[df['label'] == 1]
non_spam_df = df[df['label'] == 0]

if spam_count < non_spam_count:
    non_spam_df = resample(non_spam_df, replace=False, n_samples=spam_count, random_state=42)
else:
    spam_df = resample(spam_df, replace=True, n_samples=non_spam_count, random_state=42)

df_balanced = pd.concat([spam_df, non_spam_df])

# 학습 및 검증 데이터셋 분리
train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42)

# 데이터셋을 Hugging Face Datasets 형식으로 변환
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")

# 데이터셋 전처리
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 모델 입력에 필요한 형식으로 변환
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# 평가 지표 설정
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
    }
# 트레이너 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='epoch',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# 모델 학습
trainer.train()

# 모델 저장
model.save_pretrained('./spam_detection_model')
tokenizer.save_pretrained('./spam_detection_model')

# 예측 함수 작성
def predict_spam(input_text):
    # 입력 전처리
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 모델 예측
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).item()
    
    # 결과 반환
    if predictions == 1:
        return "스팸"
    else:
        return "스팸 아님"

# 사용자 입력 받기
user_input = input("메시지를 입력하세요: ")
result = predict_spam(user_input)
print(f"입력 문장: {user_input}\n분류 결과: {result}")