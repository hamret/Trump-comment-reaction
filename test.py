import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

data_path = "result/predicted_mainC.csv"

# 제대로 인코딩을 설정하고 컬럼 확인
df = pd.read_csv(data_path, encoding="utf-8")  # 또는 encoding="ISO-8859-1" 시도

print("✅ 컬럼명:", df.columns.tolist())  # <- 여기가 중요

# 이후 그대로
if 'text' not in df.columns:
    raise ValueError("`text` column not found in the dataset.")