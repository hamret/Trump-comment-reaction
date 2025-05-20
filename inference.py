import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm



# gpu 설정
gpu = torch.backends.mps.is_available()
device = torch.device("mps" if gpu else "cpu")
print("Using device: ", device)

# 데이터 로드
data_path = "result/predicted_mainC.csv"
df = pd.read_csv(data_path, encoding="utf-8")

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)
data_X = df['text'].tolist()
labels = df['label'].values

print(f"Number of valid samples: {len(data_X)}")

# Tokenizer 로드
tokenizer = MobileBertTokenizer.from_pretrained("donald-tariff", do_lower_case=True)

# 텍스트 데이터 토크나이징
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")

# Extract tokenized inputs
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("Tokenization complete.")

# 데이터셋 준비
batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_mask = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_mask, test_labels)
test_sampler = torch.utils.data.RandomSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
print("Dataset preparation complete.")

# 모델 로드
model = MobileBertForSequenceClassification.from_pretrained(r"/Users/daol/PycharmProjects/Trump-comment-reaction/donald-tariff")
model.to(device)

# 모델 평가 모드로 설정
model.eval()

test_pred = []
test_true = []

# 예측 수행
for batch in tqdm(test_dataloader, desc="Inferencing Full DataSet"):
    batch_ids, batch_mask, batch_labels = batch

    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)
    batch_labels = batch_labels.to(device)

    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)
    logits = output.logits
    pred = torch.argmax(logits, dim=1)
    test_pred.extend(pred.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

# 정확도 계산
test_accuracy = np.sum(np.array(test_pred) == np.array(test_true)) / len(test_pred)
print("전체 데이터 54,000건에 대한 긍부정 정확도:  ", test_accuracy)

