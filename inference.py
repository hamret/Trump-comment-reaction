import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

# GPU (MPS) 설정
gpu = torch.backends.mps.is_available()
device = torch.device("mps" if gpu else "cpu")
print(f"Using device: {device}")

# 데이터 로드
data_path = "result/predicted_traiff.csv"
try:
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.replace("ï»¿", "", regex=False)
except FileNotFoundError:
    print(f"Error: The file '{data_path}' was not found. Please check the path.")
    exit()

# 'text' 컬럼의 결측치 제거 및 문자열 타입 확인
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)

# 'label' 컬럼의 데이터 타입 확인 및 정수형으로 변환
# 레이블은 분류 모델에서 일반적으로 정수형(long)이어야 합니다.
if not pd.api.types.is_integer_dtype(df['label']):
    try:
        # float 형태 (예: 0.0, 1.0)를 정수로 변환 시도
        df['label'] = df['label'].astype(int)
        print("Labels converted to integer type.")
    except ValueError:
        print("Warning: Could not convert 'label' column to integer type. Please check label values.")
        # 변환 실패 시, 오류 방지를 위해 해당 행을 건너뛰거나 추가 처리 필요
        # 현재는 오류 발생 시 프로그램 종료 없이 진행하지만, 필요에 따라 exit() 등을 추가할 수 있습니다.

data_X = df['text'].tolist()
labels = df['label'].values

print(f"Number of valid samples: {len(data_X)}")

# 모델 및 토크나이저 경로 설정
# 로컬에 저장된 모델의 경로를 정확히 지정해주세요.
model_path = r"/Users/daol/PycharmProjects/Trump-comment-reaction/donald-tariff-finetuned"

# Tokenizer 로드 (모델과 동일한 로컬 경로 사용)
try:
    tokenizer = MobileBertTokenizer.from_pretrained(model_path, do_lower_case=True)
    print(f"Tokenizer loaded from local path: {model_path}")
except Exception as e:
    print(f"Error loading tokenizer from '{model_path}': {e}")
    print("Please ensure the path is correct and contains tokenizer files (e.g., tokenizer_config.json, vocab.txt).")
    exit()

# 텍스트 데이터 토크나이징
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
print("Tokenization complete.")

# 토크나이징된 입력 추출
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 데이터셋 준비
batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels, dtype=torch.long)  # 레이블을 torch.long 타입으로 지정
test_mask = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_mask, test_labels)

# 추론 시에는 SequentialSampler 사용 (재현성 위해)
test_sampler = torch.utils.data.SequentialSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
print("Dataset preparation complete.")

# 모델 로드 (로컬 경로 사용)
try:
    model = MobileBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    print(f"Model loaded from local path: {model_path}")
except Exception as e:
    print(f"Error loading model from '{model_path}': {e}")
    print("Please ensure the path is correct and contains model files (e.g., pytorch_model.bin).")
    exit()

# 모델 평가 모드로 설정
model.eval()

test_pred = []
test_true = []

# 예측 수행
print("Starting inference...")
for batch in tqdm(test_dataloader, desc="Inferencing Full DataSet"):
    batch_ids, batch_mask, batch_labels = batch

    # 텐서를 적절한 디바이스로 이동
    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)
    batch_labels = batch_labels.to(device)  # 레이블은 이미 torch.long 타입이므로 별도 float 변환 불필요

    with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
        output = model(batch_ids, attention_mask=batch_mask)

    logits = output.logits

    # 예측 클래스 (가장 높은 로짓 값의 인덱스)
    pred = torch.argmax(logits, dim=1)

    # 결과를 CPU로 옮겨 NumPy 배열로 변환 후 리스트에 추가
    test_pred.extend(pred.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

# 정확도 계산
test_accuracy = np.sum(np.array(test_pred) == np.array(test_true)) / len(test_pred)
print("---")
print(f"전체 데이터 {len(data_X)}건에 대한 긍부정 정확도: {test_accuracy:.4f}")