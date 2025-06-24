import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MobileBertTokenizer, MobileBertModel
from sklearn.model_selection import train_test_split
import numpy as np
import ast
from tqdm.auto import tqdm


# --- 최종 학습 스크립트의 모델/데이터셋/유틸리티 함수를 그대로 가져옵니다 ---

# 1. '개선된' 멀티레이블 MobileBERT 모델 클래스
class MultiLabelMobileBert(nn.Module):
    def __init__(self, model_name='google/mobilebert-uncased', num_labels=12, dropout_rate=0.1):
        super(MultiLabelMobileBert, self).__init__()
        self.num_labels = num_labels
        self.bert = MobileBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        return logits


# 2. 커스텀 데이터셋 클래스 (동일)
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


# 3. 제약조건 적용 함수 (최신 모델과 동일하게)
def apply_constraints(logits):
    batch_size = logits.size(0)
    # 모델 출력(0,1,2)을 원본(-1,0,1)으로 바꾸려면 bool 대신 float 사용
    predictions = torch.full_like(logits, -1, dtype=torch.float)

    for i in range(4):
        start_idx = i * 3
        end_idx = start_idx + 3
        ai_logits = logits[:, start_idx:end_idx]
        max_indices = torch.argmax(ai_logits, dim=1)  # max_indices는 0, 1, 2

        for j in range(batch_size):
            # 0,1,2 -> -1,0,1 로 변환
            predicted_label = max_indices[j].item() - 1
            predictions[j, i] = predicted_label

    # 우리는 (batch, 4) 형태의 예측이 필요
    # 이 함수는 예측 로직을 단순화하기 위해 약간 수정됨
    final_preds = torch.zeros((batch_size, 4), dtype=torch.long)
    for i in range(4):
        start_idx = i * 3
        end_idx = start_idx + 3
        final_preds[:, i] = torch.argmax(logits[:, start_idx:end_idx], dim=1) - 1

    return final_preds.cpu().numpy()


# 4. 예측 수행 함수 (최신 모델에 맞게 수정)
def get_predictions(model, data_loader, device):
    model.eval()
    predictions_list = []

    progress_bar = tqdm(data_loader, desc="Predicting", leave=True)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 제약조건을 적용하여 최종 예측을 얻음
            predictions = apply_constraints(logits)

            predictions_list.extend(predictions)

    return predictions_list


# 5. 결과 비교 및 상태 분석 함수 (동일)
def analyze_results(row):
    original = row['original_labels']
    predicted = row['model_predictions']
    mismatches = np.sum(np.array(original) != np.array(predicted))
    if mismatches == 0:
        status = 'Correct'
    elif mismatches == 4:
        status = 'Completely Wrong'
    else:
        status = 'Partially Wrong'
    return status, mismatches


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # 설정
    MODEL_NAME = 'google/mobilebert-uncased'
    DATA_PATH = 'labeled_data_only.csv'

    # --- !!! 여기가 직접 수정해야 할 부분 !!! ---
    # 가장 성능이 좋았던 최신 모델의 실제 파일 경로를 입력하세요.
    # 예: 'multilabel_mobilebert_best_20250619_101322.pt'
    BEST_MODEL_PATH = 'multilabel_mobilebert_best_20250619_101322.pt'
    # ---------------------------------------------

    OUTPUT_CSV_PATH = 'error_analysis_final_model.csv'
    MAX_LEN = 256
    BATCH_SIZE = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. '개선된' 모델 아키텍처로 로드
    print(f"Loading model architecture: {MultiLabelMobileBert.__name__}")
    model = MultiLabelMobileBert(model_name=MODEL_NAME).to(device)

    print(f"Loading model weights from: {BEST_MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{BEST_MODEL_PATH}'.")
        print("Please check the file name and path.")
        exit()

    # 2. 데이터 로드 및 검증셋 분리
    print("Loading and preparing data...")
    df = pd.read_csv(DATA_PATH)
    df['Data'] = df['Data'].fillna('')
    label_column = 'label(chatgpt,claude,grok,gemini)'
    df['original_labels'] = df[label_column].apply(ast.literal_eval)

    # 학습 때와 '동일한' random_state를 사용
    _, df_val = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Validation data size for analysis: {len(df_val)}")

    # 3. 예측용 데이터로더 생성
    tokenizer = MobileBertTokenizer.from_pretrained(MODEL_NAME)
    val_dataset = SentimentDataset(
        texts=df_val.Data.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 예측 실행
    model_predictions = get_predictions(model, val_loader, device)

    # 5. 결과 데이터프레임 생성 및 분석
    print("Analyzing results...")
    df_val['model_predictions'] = [list(p) for p in model_predictions]

    analysis_results = df_val.apply(analyze_results, axis=1, result_type='expand')
    df_val[['status', 'mismatch_count']] = analysis_results

    # 6. 최종 CSV 파일 저장
    df_final = df_val.sort_values(by='mismatch_count', ascending=False)
    output_columns = [
        'Data', 'comment', 'original_labels', 'model_predictions',
        'status', 'mismatch_count', 'Post_ID', 'Type', 'category'
    ]

    # 원본 df에 없는 컬럼이 있을 경우를 대비해 예외 처리
    final_cols_to_use = [col for col in output_columns if col in df_final.columns]

    df_final[final_cols_to_use].to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"\nAnalysis complete! Results saved to '{OUTPUT_CSV_PATH}'")