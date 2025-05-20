import torch
import pandas as pd
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm
## 훈련후 라벨링 코드
gpu = torch.backends.mps.is_available()
device = torch.device("mps" if gpu else "cpu")
print("Using device: ", device)

data_path = "testdata/mainC.csv"
df = pd.read_csv(data_path, encoding="ISO-8859-1")

if 'text' not in df.columns:
    raise ValueError("`text` column not found in the dataset.")

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)
data_X = df['text'].tolist()

print(f"Number of valid samples: {len(data_X)}")

tokenizer = MobileBertTokenizer.from_pretrained("donald-tariff", do_lower_case=True)

inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("Tokenization complete.")


batch_size = 8
test_inputs = torch.tensor(input_ids)
test_mask = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_mask)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

model = MobileBertForSequenceClassification.from_pretrained(
    "/Users/daol/PycharmProjects/Trump-comment-reaction/donald-tariff"
)
model.to(device)
model.eval()

# 예측
test_pred = []

for batch in tqdm(test_dataloader, desc="Inferencing Full DataSet"):
    batch_ids, batch_mask = batch
    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)

    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)
    logits = output.logits
    pred = torch.argmax(logits, dim=1)
    test_pred.extend(pred.cpu().numpy())

df["label"] = test_pred

output_path = "result/predicted_mainC.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"예측 결과 저장 완료: {output_path}")
