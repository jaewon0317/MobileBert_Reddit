import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import MobileBertTokenizer, MobileBertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as sklearn_f1_score, classification_report, confusion_matrix
import numpy as np
import ast
import logging
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


# 0. 평가 지표 함수들
def hamming_loss(y_true, y_pred):
    """Hamming Loss 계산: 틀린 레이블의 비율"""
    return (y_true != y_pred).mean()


def calculate_label_wise_f1(y_true, y_pred, label_names=None):
    """Label-wise F1 Score 계산"""
    if label_names is None:
        label_names = [f'Label_{i}' for i in range(y_true.shape[1])]

    f1_scores = []
    for i in range(y_true.shape[1]):
        f1 = sklearn_f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1_scores.append(f1)

    return f1_scores, label_names


def calculate_ai_wise_metrics(y_true, y_pred):
    """AI별 평균 지표 계산"""
    ai_names = ['ChatGPT', 'Claude', 'Grok', 'Gemini']
    ai_f1_scores = []
    ai_hamming_losses = []

    for ai_idx in range(4):
        start_idx = ai_idx * 3
        end_idx = start_idx + 3

        # AI별 F1 점수 (3개 감성의 평균)
        ai_true = y_true[:, start_idx:end_idx]
        ai_pred = y_pred[:, start_idx:end_idx]

        ai_f1_list = []
        for i in range(3):
            f1 = sklearn_f1_score(ai_true[:, i], ai_pred[:, i], zero_division=0)
            ai_f1_list.append(f1)

        ai_f1_scores.append(np.mean(ai_f1_list))

        # AI별 Hamming Loss
        ai_hamming_loss = hamming_loss(ai_true, ai_pred)
        ai_hamming_losses.append(ai_hamming_loss)

    return ai_f1_scores, ai_hamming_losses, ai_names


# 1. Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=0, path='best_model.pt', metric='loss'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_metric_max = -np.inf
        self.delta = delta
        self.path = path
        self.metric = metric

    def __call__(self, val_loss, val_metric, model):
        if self.metric == 'loss':
            score = -val_loss
        else:  # EMR 기준
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_metric, model):
        if self.verbose:
            if self.metric == 'loss':
                logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            else:
                logging.info(
                    f'Validation EMR increased ({self.val_metric_max:.6f} --> {val_metric:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.val_metric_max = val_metric


# 2. 멀티레이블 MobileBERT 모델
class MultiLabelMobileBert(nn.Module):
    def __init__(self, model_name='google/mobilebert-uncased', num_labels=12, dropout_rate=0.1):
        super(MultiLabelMobileBert, self).__init__()
        self.num_labels = num_labels
        self.bert = MobileBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # 더 깊은 분류기
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


# 3. 커스텀 데이터셋 클래스
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


# 4. 레이블 변환 함수
def convert_to_multilabel(labels_list):
    """[-1, 0, 1, 0] → [1,0,0, 0,1,0, 0,0,1, 0,1,0]"""
    multilabel = np.zeros(12)
    for i, label in enumerate(labels_list):
        class_idx = label + 1
        multilabel[i * 3 + class_idx] = 1
    return multilabel


# 5. 제약조건 적용 함수
def apply_constraints(logits):
    """각 AI당 하나의 감성만 선택하도록 제약 적용"""
    batch_size = logits.size(0)
    predictions = torch.zeros_like(logits, dtype=torch.bool)

    for i in range(4):  # 4개 AI
        start_idx = i * 3
        end_idx = start_idx + 3

        ai_logits = logits[:, start_idx:end_idx]
        max_indices = torch.argmax(ai_logits, dim=1)

        for j in range(batch_size):
            predictions[j, start_idx + max_indices[j]] = True

    return predictions


# 6. 개선된 손실 함수
class ConstrainedBCELoss(nn.Module):
    def __init__(self, pos_weight=None, alpha=0.1):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.alpha = alpha

    def forward(self, outputs, labels):
        # 기본 BCE Loss
        bce_loss = self.bce_loss(outputs, labels)

        # 제약조건 손실: 각 AI당 하나만 선택하도록 유도
        outputs_reshaped = outputs.view(-1, 4, 3)
        softmax_outputs = F.softmax(outputs_reshaped, dim=2)

        # 엔트로피 최소화 (더 확실한 예측 유도)
        entropy = -torch.sum(softmax_outputs * torch.log(softmax_outputs + 1e-7), dim=2)
        constraint_loss = torch.mean(entropy)

        # Softmax 출력의 합이 1에 가까워지도록
        sum_constraint = torch.mean(torch.abs(torch.sum(softmax_outputs, dim=2) - 1))

        total_loss = bce_loss - self.alpha * constraint_loss + self.alpha * sum_constraint

        return total_loss, bce_loss, constraint_loss


# 7. 개선된 학습 함수
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, gradient_clip=1.0):
    model.train()
    total_loss = 0
    total_bce_loss = 0
    total_constraint_loss = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss, bce_loss, constraint_loss = loss_fn(outputs, labels)

        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_constraint_loss += constraint_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({
            'loss': loss.item(),
            'bce': bce_loss.item(),
            'const': constraint_loss.item()
        })

    avg_losses = {
        'total': total_loss / len(data_loader),
        'bce': total_bce_loss / len(data_loader),
        'constraint': total_constraint_loss / len(data_loader)
    }

    return avg_losses


# 8. 평가 함수
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    task_correct = [0] * 4
    total_samples = 0
    exact_matches = 0

    # F1 score 계산을 위한 변수
    all_predictions = []
    all_labels = []
    task_predictions = [[] for _ in range(4)]
    task_labels = [[] for _ in range(4)]

    progress_bar = tqdm(data_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, _, _ = loss_fn(outputs, labels)
            total_loss += loss.item()

            # 제약조건 적용하여 예측
            predictions_constrained = apply_constraints(outputs)

            # 전체 예측/레이블 저장
            all_predictions.extend(predictions_constrained.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Task별 정확도 계산
            for i in range(4):
                start_idx = i * 3
                end_idx = start_idx + 3

                pred_ai = predictions_constrained[:, start_idx:end_idx]
                true_ai = labels[:, start_idx:end_idx]

                pred_classes = torch.argmax(pred_ai.float(), dim=1)
                true_classes = torch.argmax(true_ai, dim=1)

                task_predictions[i].extend(pred_classes.cpu().numpy())
                task_labels[i].extend(true_classes.cpu().numpy())

                task_correct[i] += torch.sum(pred_classes == true_classes).item()

            # Exact Match
            exact_matches += torch.sum(
                torch.all(predictions_constrained == labels.bool(), dim=1)
            ).item()

            total_samples += labels.size(0)

    # 지표 계산
    avg_loss = total_loss / len(data_loader)
    accuracies = [tc / total_samples for tc in task_correct]
    exact_match_ratio = exact_matches / total_samples

    # F1 scores
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    f1_micro = sklearn_f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    f1_macro = sklearn_f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    # Task별 F1 scores
    task_f1_scores = []
    for i in range(4):
        f1 = sklearn_f1_score(task_labels[i], task_predictions[i], average='macro', zero_division=0)
        task_f1_scores.append(f1)

    # 1. Hamming Loss 계산
    hamming_loss_value = hamming_loss(all_labels.astype(int), all_predictions.astype(int))

    # 2. Label-wise F1 Score 계산
    label_names = ['ChatGPT_neg', 'ChatGPT_neu', 'ChatGPT_pos',
                   'Claude_neg', 'Claude_neu', 'Claude_pos',
                   'Grok_neg', 'Grok_neu', 'Grok_pos',
                   'Gemini_neg', 'Gemini_neu', 'Gemini_pos']

    label_f1_scores, _ = calculate_label_wise_f1(all_labels.astype(int), all_predictions.astype(int), label_names)

    # 3. AI별 지표 계산
    ai_f1_scores, ai_hamming_losses, ai_names = calculate_ai_wise_metrics(all_labels.astype(int),
                                                                          all_predictions.astype(int))

    metrics = {
        'loss': avg_loss,
        'accuracies': accuracies,
        'emr': exact_match_ratio,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'task_f1_scores': task_f1_scores,
        'predictions': (task_predictions, task_labels),
        # 새로 추가된 지표들
        'hamming_loss': hamming_loss_value,
        'label_f1_scores': label_f1_scores,
        'label_names': label_names,
        'ai_f1_scores': ai_f1_scores,
        'ai_hamming_losses': ai_hamming_losses,
        'ai_names': ai_names
    }

    return metrics


# 9. Confusion Matrix 시각화 함수
def plot_confusion_matrices(task_predictions, task_labels, task_names, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (preds, labels, name) in enumerate(zip(task_predictions, task_labels, task_names)):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 10. 메인 실행 부분
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    MODEL_NAME = 'google/mobilebert-uncased'
    DATA_PATH = 'labeled_data_only.csv'
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = f'multilabel_training_{TIMESTAMP}.log'
    BEST_MODEL_PATH = f'multilabel_mobilebert_best_{TIMESTAMP}.pt'
    MAX_LEN = 256
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 2e-5  # 감소
    PATIENCE = 7  # 증가
    DROPOUT_RATE = 0.1
    GRADIENT_CLIP = 1.0
    CONSTRAINT_ALPHA = 0.1

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

    # 텐서보드 Writer 초기화
    writer = SummaryWriter(f'runs/mobilebert_multilabel_{TIMESTAMP}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Max length: {MAX_LEN}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    logging.info(f"Constraint alpha: {CONSTRAINT_ALPHA}, Gradient clip: {GRADIENT_CLIP}")

    # 데이터 로드 및 전처리
    logging.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['Data'] = df['Data'].fillna('')

    label_column = 'label(chatgpt,claude,grok,gemini)'
    df['labels_list'] = df[label_column].apply(ast.literal_eval)
    df['multilabel'] = df['labels_list'].apply(convert_to_multilabel)

    # 카테고리 분포 확인
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        logging.info(f"Category distribution:\n{category_counts}")

    # 데이터 분할 (stratify 옵션 처리)
    try:
        df_train, df_val = train_test_split(df, test_size=0.15, random_state=42, stratify=df['category'])
        logging.info("Using stratified split by category")
    except ValueError as e:
        logging.warning(f"Stratified split failed: {e}")
        logging.info("Falling back to random split without stratification")
        df_train, df_val = train_test_split(df, test_size=0.15, random_state=42)

    logging.info(f"Train data size: {len(df_train)}, Validation data size: {len(df_val)}")

    # 클래스 분포 로깅
    train_labels = np.stack(df_train['multilabel'].values)
    val_labels = np.stack(df_val['multilabel'].values)
    logging.info(f"Train label distribution: {train_labels.sum(axis=0)}")
    logging.info(f"Validation label distribution: {val_labels.sum(axis=0)}")

    # 토크나이저, 데이터셋, 데이터로더 생성
    tokenizer = MobileBertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SentimentDataset(
        texts=df_train['Data'].values,
        labels=df_train['multilabel'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = SentimentDataset(
        texts=df_val['Data'].values,
        labels=df_val['multilabel'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # 모델 초기화
    model = MultiLabelMobileBert(model_name=MODEL_NAME, dropout_rate=DROPOUT_RATE).to(device)

    # 클래스 가중치 계산
    pos_weight = torch.tensor([
        min(10.0, max(0.1, (1 - train_labels[:, i].mean()) / (train_labels[:, i].mean() + 1e-7)))
        for i in range(12)
    ]).to(device)

    logging.info(f"Positive weights: {pos_weight}")

    # 손실함수, 옵티마이저, 스케줄러
    loss_fn = ConstrainedBCELoss(pos_weight=pos_weight, alpha=CONSTRAINT_ALPHA).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Early stopping (EMR 기준)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=BEST_MODEL_PATH, metric='emr')

    # 학습 시작
    task_names = ['ChatGPT', 'Claude', 'Grok', 'Gemini']
    best_emr = 0.0

    logging.info("Starting training...")
    logging.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(EPOCHS):
        logging.info(f'\n{"=" * 50}')
        logging.info(f'Epoch {epoch + 1}/{EPOCHS}')
        logging.info(f'{"=" * 50}')

        # 학습
        train_losses = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, GRADIENT_CLIP)
        logging.info(f'Train losses - Total: {train_losses["total"]:.6f}, '
                     f'BCE: {train_losses["bce"]:.6f}, Constraint: {train_losses["constraint"]:.6f}')

        # 평가
        val_metrics = eval_model(model, val_loader, loss_fn, device)

        # 텐서보드 기록
        writer.add_scalars('Loss', {
            'train_total': train_losses["total"],
            'train_bce': train_losses["bce"],
            'train_constraint': train_losses["constraint"],
            'validation': val_metrics["loss"]
        }, epoch)

        writer.add_scalars('Accuracy/Individual_Tasks', {
            task_names[i]: val_metrics["accuracies"][i] for i in range(4)
        }, epoch)

        writer.add_scalars('F1_Score/Individual_Tasks', {
            task_names[i]: val_metrics["task_f1_scores"][i] for i in range(4)
        }, epoch)

        writer.add_scalar('Metrics/Average_Accuracy', np.mean(val_metrics["accuracies"]), epoch)
        writer.add_scalar('Metrics/Exact_Match_Ratio', val_metrics["emr"], epoch)
        writer.add_scalar('Metrics/F1_Micro', val_metrics["f1_micro"], epoch)
        writer.add_scalar('Metrics/F1_Macro', val_metrics["f1_macro"], epoch)

        # === 새로 추가된 텐서보드 로깅 ===
        writer.add_scalar('Metrics/Hamming_Loss', val_metrics["hamming_loss"], epoch)

        # AI별 F1과 Hamming Loss 로깅
        writer.add_scalars('AI_F1_Scores', {
            val_metrics["ai_names"][i]: val_metrics["ai_f1_scores"][i] for i in range(4)
        }, epoch)

        writer.add_scalars('AI_Hamming_Loss', {
            val_metrics["ai_names"][i]: val_metrics["ai_hamming_losses"][i] for i in range(4)
        }, epoch)

        # 로깅
        logging.info(f'Validation loss: {val_metrics["loss"]:.6f}')
        logging.info(f'\nTask-wise Performance:')
        for i, name in enumerate(task_names):
            logging.info(
                f'  - {name}: Acc={val_metrics["accuracies"][i]:.4f} ({val_metrics["accuracies"][i] * 100:.2f}%), '
                f'F1={val_metrics["task_f1_scores"][i]:.4f}')

        logging.info(f'\nOverall Performance:')
        logging.info(
            f'  - Average Accuracy: {np.mean(val_metrics["accuracies"]):.4f} ({np.mean(val_metrics["accuracies"]) * 100:.2f}%)')
        logging.info(f'  - Exact Match Ratio (EMR): {val_metrics["emr"]:.4f} ({val_metrics["emr"] * 100:.2f}%)')
        logging.info(f'  - F1 Score (Micro): {val_metrics["f1_micro"]:.4f}')
        logging.info(f'  - F1 Score (Macro): {val_metrics["f1_macro"]:.4f}')

        # === 새로 추가된 로깅 ===
        logging.info(f'  - Hamming Loss: {val_metrics["hamming_loss"]:.4f}')

        logging.info(f'\nAI-wise Performance:')
        for i, ai_name in enumerate(val_metrics["ai_names"]):
            logging.info(
                f'  - {ai_name}: F1={val_metrics["ai_f1_scores"][i]:.4f}, '
                f'Hamming Loss={val_metrics["ai_hamming_losses"][i]:.4f}')

        logging.info(f'\nLabel-wise F1 Scores:')
        for label_name, f1_score in zip(val_metrics["label_names"], val_metrics["label_f1_scores"]):
            logging.info(f'  - {label_name}: {f1_score:.4f}')

        # Best EMR 업데이트
        if val_metrics["emr"] > best_emr:
            best_emr = val_metrics["emr"]
            logging.info(f'  - New best EMR achieved!')

            # Confusion matrices 저장
            task_preds, task_true = val_metrics["predictions"]
            plot_confusion_matrices(task_preds, task_true, task_names,
                                    f'confusion_matrices_epoch{epoch + 1}.png')

        # Early stopping (EMR 기준)
        early_stopping(val_metrics["loss"], val_metrics["emr"], model)
        if early_stopping.early_stop:
            logging.info("\nEarly stopping triggered!")
            break

    writer.close()

    # 최종 평가
    logging.info(f'\n{"=" * 50}')
    logging.info("Training complete!")
    logging.info(f'{"=" * 50}')
    logging.info(f"Best Exact Match Ratio: {best_emr:.4f} ({best_emr * 100:.2f}%)")
    logging.info(f"Loading best model from {BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    logging.info("\nFinal evaluation on validation set:")
    val_metrics = eval_model(model, val_loader, loss_fn, device)

    logging.info(f'\nFinal Results:')
    for i, name in enumerate(task_names):
        logging.info(
            f'  - {name}: Accuracy={val_metrics["accuracies"][i]:.4f} ({val_metrics["accuracies"][i] * 100:.2f}%), '
            f'F1={val_metrics["task_f1_scores"][i]:.4f}')

    logging.info(f'\nOverall Metrics:')
    logging.info(
        f'  - Average Accuracy: {np.mean(val_metrics["accuracies"]):.4f} ({np.mean(val_metrics["accuracies"]) * 100:.2f}%)')
    logging.info(f'  - Exact Match Ratio: {val_metrics["emr"]:.4f} ({val_metrics["emr"] * 100:.2f}%)')
    logging.info(f'  - F1 Score (Micro): {val_metrics["f1_micro"]:.4f}')
    logging.info(f'  - F1 Score (Macro): {val_metrics["f1_macro"]:.4f}')
    logging.info(f'  - Hamming Loss: {val_metrics["hamming_loss"]:.4f}')

    logging.info(f'\nFinal AI-wise Performance:')
    for i, ai_name in enumerate(val_metrics["ai_names"]):
        logging.info(
            f'  - {ai_name}: F1={val_metrics["ai_f1_scores"][i]:.4f}, '
            f'Hamming Loss={val_metrics["ai_hamming_losses"][i]:.4f}')

    logging.info(f'\nFinal Label-wise F1 Scores:')
    for label_name, f1_score in zip(val_metrics["label_names"], val_metrics["label_f1_scores"]):
        logging.info(f'  - {label_name}: {f1_score:.4f}')

    # 최종 confusion matrices 저장
    task_preds, task_true = val_metrics["predictions"]
    plot_confusion_matrices(task_preds, task_true, task_names, f'final_confusion_matrices_{TIMESTAMP}.png')

    # 클래스별 분포 출력
    logging.info("\nPer-task class distribution (Predicted):")
    for i, name in enumerate(task_names):
        pred_counts = np.bincount(task_preds[i], minlength=3)
        logging.info(f'  - {name}: Negative={pred_counts[0]}, Neutral={pred_counts[1]}, Positive={pred_counts[2]}')

    # === 추가 분석 및 요약 ===
    logging.info(f'\n{"=" * 50}')
    logging.info("FINAL SUMMARY")
    logging.info(f'{"=" * 50}')

    # 가장 성능이 좋은/나쁜 AI 찾기
    best_ai_idx = np.argmax(val_metrics["ai_f1_scores"])
    worst_ai_idx = np.argmin(val_metrics["ai_f1_scores"])

    logging.info(
        f"Best performing AI: {val_metrics['ai_names'][best_ai_idx]} (F1: {val_metrics['ai_f1_scores'][best_ai_idx]:.4f})")
    logging.info(
        f"Worst performing AI: {val_metrics['ai_names'][worst_ai_idx]} (F1: {val_metrics['ai_f1_scores'][worst_ai_idx]:.4f})")

    # 가장 성능이 좋은/나쁜 레이블 찾기
    best_label_idx = np.argmax(val_metrics["label_f1_scores"])
    worst_label_idx = np.argmin(val_metrics["label_f1_scores"])

    logging.info(
        f"Best performing label: {val_metrics['label_names'][best_label_idx]} (F1: {val_metrics['label_f1_scores'][best_label_idx]:.4f})")
    logging.info(
        f"Worst performing label: {val_metrics['label_names'][worst_label_idx]} (F1: {val_metrics['label_f1_scores'][worst_label_idx]:.4f})")

    # 전체 성능 요약
    overall_accuracy = np.mean(val_metrics["accuracies"])
    overall_f1 = np.mean(val_metrics["ai_f1_scores"])

    logging.info(f"\nOverall Performance Summary:")
    logging.info(f"  - Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    logging.info(f"  - Overall F1 Score: {overall_f1:.4f}")
    logging.info(f"  - Exact Match Ratio: {val_metrics['emr']:.4f} ({val_metrics['emr'] * 100:.2f}%)")
    logging.info(f"  - Hamming Loss: {val_metrics['hamming_loss']:.4f}")

    logging.info(f"\nModel and logs saved. Training session completed at {datetime.now()}")