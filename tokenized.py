import pandas as pd
from transformers import MobileBertTokenizer
import json
from tqdm import tqdm

# 1. 데이터 로드
path = "testdata/mainC_labeled_f.csv"
try:
    df = pd.read_csv(path, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(path, encoding="cp949")
    except Exception as e:
        print(f"CSV 파일 로드 오류: {e}")
        exit()
except Exception as e:
    print(f"CSV 파일 로드 오류: {e}")
    exit()

# 필요한 열 확인 및 'text' 열 지정
text_column = 'text'
if 'comment_text' in df.columns:
    text_column = 'comment_text'
elif 'text' not in df.columns:
    print(f"오류: '{text_column}' 열이 DataFrame에 없습니다.")
    exit()

# 데이터 개수 확인
num_rows = len(df)
print(f"로드된 데이터 행 개수: {num_rows}")
if num_rows != 52990:
    print(f"경고: 예상 행 개수(52990)와 실제 행 개수({num_rows})가 다릅니다.")

# 2. 데이터 전처리: 결측값 제거 및 강제 형 변환
# 결측값을 빈 문자열로 채웁니다.
df[text_column] = df[text_column].fillna("")
# 모든 값을 문자열로 변환
df[text_column] = df[text_column].astype(str)

# 3. 토크나이저 불러오기
tokenizer = MobileBertTokenizer.from_pretrained('mobliebert-uncased', do_lower_case=True)
MAX_LEN = 256


def tokenize_texts(texts):
    # 리스트 형식으로 입력받아 처리
    return tokenizer(
        texts, truncation=True, max_length=MAX_LEN,
        padding='max_length', return_attention_mask=True, return_tensors='pt'
    )


# 4. 'text' 열의 데이터 토큰화
all_texts = df[text_column].tolist()
tokenized_data = []

for text in tqdm(all_texts, desc="토큰화 진행"):
    try:
        inputs = tokenize_texts([text])  # 각 행을 개별적으로 토큰화
        tokenized_data.append({
            'input_ids': inputs['input_ids'].tolist()[0],
            'attention_mask': inputs['attention_mask'].tolist()[0]
        })
    except Exception as e:
        # 에러 발생 시 로그 출력
        print(f"토큰화 오류: {e}. text: {text}")
        continue

# 5. 토큰화된 데이터를 JSON 파일로 저장
output_path = "tokenized_final.json"
with open(output_path, 'w') as f:
    json.dump(tokenized_data, f)

print(f"\n'{text_column}' 열의 데이터가 {output_path}에 토큰화되어 저장되었습니다.")
print(f"토큰화된 데이터 개수: {len(tokenized_data)}")
