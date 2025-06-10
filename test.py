import pandas as pd

data_path = "result/predicted_traiff.csv"
try:
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    print(df.columns)
except FileNotFoundError:
    print(f"파일이 존재하지 않습니다: {data_path}")
except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")

