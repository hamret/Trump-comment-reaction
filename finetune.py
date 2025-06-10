import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device: ", device)

# transformers 라이브러리 로깅 레벨 설정 (에러만 표시)
logging.set_verbosity_error()

# 데이터 로드
path = "maindata/trump_tariff_2000.csv"
df = pd.read_csv(path, encoding="ISO-8859-1")  # 인코딩 문제 해결

# 결측치 제거 및 텍스트 형변환
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)

# 데이터 준비
data_X = list(df['text'].values)
labels = df['label'].values

# 데이터 샘플 출력
print("데이터 샘플:")
print("리뷰문장: ", data_X[:3])
print("지지/비지지/중립: ", labels[:3])

# MobileBERT tokenizer 사용
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors="pt")

# input_ids, attention_mask 추출
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# ⚠️ label을 반드시 long 타입으로 변환 (여기서 에러가 발생했었음!)
labels = torch.tensor(labels, dtype=torch.long)

# 디버그용: 라벨 데이터 타입 확인
print("라벨 값 데이터 타입 확인:", labels.dtype)
print("라벨 값 종류:", np.unique(labels))

# 데이터셋 분할
train, val, train_y, val_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

# DataLoader 설정
batch_size = 8
train_dataset = TensorDataset(train, train_mask, train_y)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

val_dataset = TensorDataset(val, val_mask, val_y)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# MobileBERT 모델 로드 및 설정
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=3)
model.config.problem_type = "single_label_classification"  # CrossEntropyLoss 사용하도록 설정
model.to(device)

# 옵티마이저 및 스케줄러 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

# 학습 loop
epochs_results = []
for e in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_dataloader, desc=f"Training Epoch {e + 1}")
    for batch in loop:
        input_id, mask, label = [b.to(device) for b in batch]
        model.zero_grad()
        output = model(input_id, attention_mask=mask, labels=label)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)

    # 평가 함수
    def evaluate(dataloader):
        model.eval()
        predictions, true_labels = [], []
        for batch in dataloader:
            input_id, mask, label = [b.to(device) for b in batch]
            with torch.no_grad():
                output = model(input_id, attention_mask=mask)
            logits = output.logits
            preds = torch.argmax(logits, axis=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(label.cpu().numpy())
        acc = np.sum(np.array(predictions) == np.array(true_labels)) / len(predictions)
        return acc

    # train/val 평가
    train_acc = evaluate(train_dataloader)
    val_acc = evaluate(val_dataloader)
    epochs_results.append((avg_train_loss, train_acc, val_acc))

# 결과 출력
for idx, (loss, train_acc, val_acc) in enumerate(epochs_results, start=1):
    print(f"Epoch {idx}: Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# 모델 저장
print("\n## 모델 저장 ##")
save_path = "donald-tariff-finetuned"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("모델 저장 완료:", save_path)
