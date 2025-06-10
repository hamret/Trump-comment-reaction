import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

gpu = torch.backends.mps.is_available()
device = torch.device("mps" if gpu else "cpu")
print(f"Using device: {device}")

data_path = "result/predicted_traiff.csv"
try:
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.replace("ï»¿", "", regex=False)
except FileNotFoundError:
    print(f"Error: The file '{data_path}' was not found. Please check the path.")
    exit()

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)

if not pd.api.types.is_integer_dtype(df['label']):
    try:

        df['label'] = df['label'].astype(int)
        print("Labels converted to integer type.")
    except ValueError:
        print("Warning: Could not convert 'label' column to integer type. Please check label values.")


data_X = df['text'].tolist()
labels = df['label'].values

print(f"Number of valid samples: {len(data_X)}")

model_path = r"/Users/daol/PycharmProjects/Trump-comment-reaction/donald-tariff-finetuned"

try:
    tokenizer = MobileBertTokenizer.from_pretrained(model_path, do_lower_case=True)
    print(f"Tokenizer loaded from local path: {model_path}")
except Exception as e:
    print(f"Error loading tokenizer from '{model_path}': {e}")
    print("Please ensure the path is correct and contains tokenizer files (e.g., tokenizer_config.json, vocab.txt).")
    exit()

inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
print("Tokenization complete.")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels, dtype=torch.long)
test_mask = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_mask, test_labels)

test_sampler = torch.utils.data.SequentialSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
print("Dataset preparation complete.")

try:
    model = MobileBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    print(f"Model loaded from local path: {model_path}")
except Exception as e:
    print(f"Error loading model from '{model_path}': {e}")
    print("Please ensure the path is correct and contains model files (e.g., pytorch_model.bin).")
    exit()

model.eval()

test_pred = []
test_true = []

print("Starting inference...")
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
print("---")
print(f"전체 데이터 {len(data_X)}건에 대한 긍부정 정확도: {test_accuracy:.4f}")