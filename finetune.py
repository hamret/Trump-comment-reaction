import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device: ", device)

logging.set_verbosity_error()

path = "data/mainC_labeled_f.csv"
df = pd.read_csv(path, encoding="utf-8")

# 결측값 제거 및 데이터 정리
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)

data_X = list(df['text'].values)
labels = df['label'].values

print("데이터 샘플:")
print("리뷰문장: ", data_X[:3])
print("긍정/부정: ", labels[:3])

tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)

inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors="pt")

input_ids = inputs['input_ids'].tolist()
attention_mask = inputs['attention_mask'].tolist()

train, val, train_y, val_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

batch_size = 8
train_dataset = TensorDataset(torch.tensor(train), torch.tensor(train_mask), torch.tensor(train_y))
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

val_dataset = TensorDataset(torch.tensor(val), torch.tensor(val_mask), torch.tensor(val_y))
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# 모델 로드 및 설정
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

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

    train_acc = evaluate(train_dataloader)
    val_acc = evaluate(val_dataloader)
    epochs_results.append((avg_train_loss, train_acc, val_acc))

for idx, (loss, train_acc, val_acc) in enumerate(epochs_results, start=1):
    print(f"Epoch {idx}: Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

print("\n## 모델 저장 ##")
save_path = "donald-comment"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("모델 저장 완료:", save_path)

