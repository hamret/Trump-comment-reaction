import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

# gpu = torch.cuda.is_available()
gpu = torch.backends.mps.is_available()

device = torch.device("mps" if gpu else "cpu")
print("Using device: ", device)


data_path = "data/mainC_labeled_f.csv"
df = pd.read_csv(data_path, encoding="utf-8")


if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("`text` or `label` column not found in the dataset.")

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)
data_X = df['text'].tolist()
labels = df['label'].values

print(f"Number of valid samples: {len(data_X)}")


tokenizers = MobileBertTokenizer.from_pretrained("donald-comment", do_lower_case=True)
try:
    inputs = tokenizers(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
except ValueError as e:
    print(f"Tokenizer error: {e}")
    raise ValueError("Please check the input data format. Ensure all entries in `data_X` are strings.")

# Extract tokenized inputs
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("Tokenization complete.")


batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_mask = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_mask, test_labels)
test_sampler = torch.utils.data.RandomSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
print("Dataset preparation complete.")


model = MobileBertForSequenceClassification.from_pretrained(r"/Users/daol/PycharmProjects/Trump-comment-reaction/donald-comment")
model.to(device)


model.eval()

test_pred = []
test_true = []

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

test_accuracy = np.sum(np.array(test_pred) == np.array(test_true)) / len(test_pred)

print("전체 데이터 54,000건에 대한 긍부정 정확도:  ", test_accuracy)
