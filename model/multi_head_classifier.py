import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
import torch.nn as nn
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


# 0. 추가된 평가 지표 함수들
def calculate_per_class_metrics(y_true, y_pred, task_names):
    """각 AI의 클래스별(감성별) 성능 계산"""
    per_class_metrics = {}

    for i, task_name in enumerate(task_names):
        # 각 AI의 confusion matrix 계산
        cm = confusion_matrix(y_true[i], y_pred[i], labels=[-1, 0, 1])

        # 클래스별 정밀도, 재현율 계산
        class_metrics = {}
        class_names = ['Negative', 'Neutral', 'Positive']

        for j, class_name in enumerate(class_names):
            # True Positives, False Positives, False Negatives
            tp = cm[j, j]
            fp = cm[:, j].sum() - tp
            fn = cm[j, :].sum() - tp

            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': cm[j, :].sum()
            }

        per_class_metrics[task_name] = class_metrics

    return per_class_metrics


# 1. Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, path='best_model.pt', metric='emr'):
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


# 2. Multi-Head MobileBERT 모델
class MultiHeadMobileBert(nn.Module):
    def __init__(self, model_name='google/mobilebert-uncased', num_tasks=4, num_classes_per_task=3, dropout_rate=0.1):
        super(MultiHeadMobileBert, self).__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.bert = MobileBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        #분류기
        hidden_size = self.bert.config.hidden_size
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes_per_task)
            ) for _ in range(num_tasks)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits_list = [classifier(pooler_output) for classifier in self.classifiers]
        return logits_list


# 3. 커스텀 데이터셋 클래스
class SentimentDataset4D(Dataset):
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
            'labels': torch.tensor(label, dtype=torch.long) + 1  # [-1, 0, 1] -> [0, 1, 2]
        }


# 4. 학습 함수
def train_epoch(model, data_loader, loss_fns, optimizer, device, scheduler, gradient_clip=1.0):
    model.train()
    total_loss = 0
    task_losses = [0] * model.num_tasks

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits_list = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = 0
        batch_task_losses = []
        for i in range(model.num_tasks):
            task_loss = loss_fns[i](logits_list[i], labels[:, i])
            loss += task_loss
            task_losses[i] += task_loss.item()
            batch_task_losses.append(task_loss.item())

        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({
            'loss': loss.item(),
            'ChatGPT': f'{batch_task_losses[0]:.3f}',
            'Claude': f'{batch_task_losses[1]:.3f}',
            'Grok': f'{batch_task_losses[2]:.3f}',
            'Gemini': f'{batch_task_losses[3]:.3f}'
        })

    avg_losses = {
        'total': total_loss / len(data_loader),
        'tasks': [loss / len(data_loader) for loss in task_losses]
    }

    return avg_losses


# 5. 평가 함수
def eval_model(model, data_loader, loss_fns, device):
    model.eval()
    total_loss = 0
    task_losses = [0] * model.num_tasks
    task_correct = [0] * model.num_tasks
    total_samples = 0
    exact_matches = 0

    # 예측과 레이블 저장
    all_predictions = []
    all_labels = []
    task_predictions = [[] for _ in range(model.num_tasks)]
    task_labels = [[] for _ in range(model.num_tasks)]

    progress_bar = tqdm(data_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits_list = model(input_ids=input_ids, attention_mask=attention_mask)

            # 손실 계산
            loss = 0
            for i in range(model.num_tasks):
                task_loss = loss_fns[i](logits_list[i], labels[:, i])
                loss += task_loss
                task_losses[i] += task_loss.item()

            total_loss += loss.item()

            # 예측 생성
            preds_per_batch = torch.stack([torch.argmax(logits, dim=1) for logits in logits_list], dim=1)

            # 전체 예측/레이블 저장 (클래스 인덱스를 원래 값으로 변환: [0,1,2] -> [-1,0,1])
            batch_preds_original = preds_per_batch - 1
            batch_labels_original = labels - 1

            all_predictions.extend(batch_preds_original.cpu().numpy())
            all_labels.extend(batch_labels_original.cpu().numpy())

            # Task별 정확도 계산
            for i in range(model.num_tasks):
                pred_classes = preds_per_batch[:, i]
                true_classes = labels[:, i]

                task_predictions[i].extend((pred_classes - 1).cpu().numpy())
                task_labels[i].extend((true_classes - 1).cpu().numpy())

                task_correct[i] += torch.sum(pred_classes == true_classes).item()

            # Exact Match
            exact_matches += torch.sum(
                torch.all(preds_per_batch == labels, dim=1)
            ).item()

            total_samples += labels.size(0)

    # 지표 계산
    avg_loss = total_loss / len(data_loader)
    avg_task_losses = [loss / len(data_loader) for loss in task_losses]
    accuracies = [tc / total_samples for tc in task_correct]
    exact_match_ratio = exact_matches / total_samples

    # 배열로 변환
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # F1 scores
    task_f1_scores = []
    for i in range(model.num_tasks):
        f1 = sklearn_f1_score(task_labels[i], task_predictions[i], average='macro', zero_division=0)
        task_f1_scores.append(f1)

    # 클래스별 성능 지표 계산
    per_class_metrics = calculate_per_class_metrics(task_predictions, task_labels, task_names)

    metrics = {
        'loss': avg_loss,
        'task_losses': avg_task_losses,
        'accuracies': accuracies,
        'emr': exact_match_ratio,
        'task_f1_scores': task_f1_scores,
        'predictions': (task_predictions, task_labels),
        'per_class_metrics': per_class_metrics
    }

    return metrics


# 6. Confusion Matrix 시각화 함수
def plot_confusion_matrices(task_predictions, task_labels, task_names, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (preds, labels, name) in enumerate(zip(task_predictions, task_labels, task_names)):
        cm = confusion_matrix(labels, preds, labels=[-1, 0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 7. 메인 실행 부분
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    MODEL_NAME = 'google/mobilebert-uncased'
    DATA_PATH = 'labeled_data_only.csv'
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = f'multihead_training_{TIMESTAMP}.log'
    BEST_MODEL_PATH = f'multihead_mobilebert_best_{TIMESTAMP}.pt'
    MAX_LEN = 256
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 2e-5
    PATIENCE = 7
    DROPOUT_RATE = 0.1
    GRADIENT_CLIP = 1.0

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
    writer = SummaryWriter(f'runs/multihead_mobilebert_{TIMESTAMP}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Max length: {MAX_LEN}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    logging.info(f"Dropout rate: {DROPOUT_RATE}, Gradient clip: {GRADIENT_CLIP}")

    # 데이터 로드 및 전처리
    logging.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['Data'] = df['Data'].fillna('')

    label_column = 'label(chatgpt,claude,grok,gemini)'
    df['labels_list'] = df[label_column].apply(ast.literal_eval)

    # 카테고리 분포 확인
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        logging.info(f"Category distribution:\n{category_counts}")

    # 데이터 분할
    try:
        df_train, df_val = train_test_split(df, test_size=0.15, random_state=42, stratify=df['category'])
        logging.info("Using stratified split by category")
    except ValueError as e:
        logging.warning(f"Stratified split failed: {e}")
        logging.info("Falling back to random split without stratification")
        df_train, df_val = train_test_split(df, test_size=0.15, random_state=42)

    logging.info(f"Train data size: {len(df_train)}, Validation data size: {len(df_val)}")

    # 클래스 분포 로깅
    train_labels = np.array([np.array(x) for x in df_train['labels_list'].values])
    val_labels = np.array([np.array(x) for x in df_val['labels_list'].values])

    logging.info("Train label distribution per task:")
    task_names = ['ChatGPT', 'Claude', 'Grok', 'Gemini']
    for i, name in enumerate(task_names):
        unique, counts = np.unique(train_labels[:, i], return_counts=True)
        dist_str = ", ".join([f"{val}: {cnt}" for val, cnt in zip(unique, counts)])
        logging.info(f"  - {name}: {dist_str}")

    # 토크나이저, 데이터셋, 데이터로더 생성
    tokenizer = MobileBertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SentimentDataset4D(
        texts=df_train['Data'].values,
        labels=df_train['labels_list'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = SentimentDataset4D(
        texts=df_val['Data'].values,
        labels=df_val['labels_list'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # 모델 초기화
    model = MultiHeadMobileBert(model_name=MODEL_NAME, dropout_rate=DROPOUT_RATE).to(device)

    # 클래스 불균형 처리를 위한 가중치 계산
    train_labels_shifted = train_labels + 1  # [-1, 0, 1] -> [0, 1, 2]
    loss_fns = []

    for i in range(4):
        class_counts = np.bincount(train_labels_shifted[:, i], minlength=3)
        class_weights = 1. / (torch.tensor(class_counts, dtype=torch.float) + 1e-7)
        class_weights = class_weights / class_weights.sum() * 3  # Normalize
        loss_fns.append(nn.CrossEntropyLoss(weight=class_weights.to(device)))
        logging.info(f"Task {i + 1} ({task_names[i]}) Class Weights: {class_weights.numpy()}")

    # 옵티마이저, 스케줄러
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
    best_emr = 0.0

    logging.info("Starting training...")
    logging.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(EPOCHS):
        logging.info(f'\n{"=" * 50}')
        logging.info(f'Epoch {epoch + 1}/{EPOCHS}')
        logging.info(f'{"=" * 50}')

        # 학습
        train_losses = train_epoch(model, train_loader, loss_fns, optimizer, device, scheduler, GRADIENT_CLIP)
        logging.info(f'Train losses - Total: {train_losses["total"]:.6f}')
        logging.info(f'Task-wise train losses: ' +
                     ', '.join([f'{task_names[i]}: {train_losses["tasks"][i]:.4f}' for i in range(4)]))

        # 평가
        val_metrics = eval_model(model, val_loader, loss_fns, device)

        # 텐서보드 기록
        writer.add_scalars('Loss', {
            'train_total': train_losses["total"],
            'validation': val_metrics["loss"]
        }, epoch)

        writer.add_scalars('Loss/Train_Tasks', {
            task_names[i]: train_losses["tasks"][i] for i in range(4)
        }, epoch)

        writer.add_scalars('Loss/Val_Tasks', {
            task_names[i]: val_metrics["task_losses"][i] for i in range(4)
        }, epoch)

        writer.add_scalars('Accuracy/Individual_Tasks', {
            task_names[i]: val_metrics["accuracies"][i] for i in range(4)
        }, epoch)

        writer.add_scalars('F1_Score/Individual_Tasks', {
            task_names[i]: val_metrics["task_f1_scores"][i] for i in range(4)
        }, epoch)

        writer.add_scalar('Metrics/Average_Accuracy', np.mean(val_metrics["accuracies"]), epoch)
        writer.add_scalar('Metrics/Exact_Match_Ratio', val_metrics["emr"], epoch)

        # 클래스별 F1 텐서보드 기록
        for task_name, class_metrics in val_metrics["per_class_metrics"].items():
            for class_name, metrics in class_metrics.items():
                writer.add_scalar(f'Class_F1/{task_name}/{class_name}', metrics['f1'], epoch)

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

        # 클래스별 성능 요약 (간단히)
        logging.info(f'\nPer-class F1 scores (summary):')
        for task_name in task_names:
            class_f1s = [val_metrics["per_class_metrics"][task_name][cls]['f1']
                         for cls in ['Negative', 'Neutral', 'Positive']]
            logging.info(f'  - {task_name}: Neg={class_f1s[0]:.3f}, Neu={class_f1s[1]:.3f}, Pos={class_f1s[2]:.3f}')

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
    val_metrics = eval_model(model, val_loader, loss_fns, device)

    logging.info(f'\nFinal Results:')
    for i, name in enumerate(task_names):
        logging.info(
            f'  - {name}: Accuracy={val_metrics["accuracies"][i]:.4f} ({val_metrics["accuracies"][i] * 100:.2f}%), '
            f'F1={val_metrics["task_f1_scores"][i]:.4f}')

    logging.info(f'\nOverall Metrics:')
    logging.info(
        f'  - Average Accuracy: {np.mean(val_metrics["accuracies"]):.4f} ({np.mean(val_metrics["accuracies"]) * 100:.2f}%)')
    logging.info(f'  - Exact Match Ratio: {val_metrics["emr"]:.4f} ({val_metrics["emr"] * 100:.2f}%)')

    # 상세한 클래스별 성능
    logging.info(f'\nDetailed per-class performance:')
    for task_name in task_names:
        logging.info(f'\n  {task_name}:')
        for class_name in ['Negative', 'Neutral', 'Positive']:
            metrics = val_metrics["per_class_metrics"][task_name][class_name]
            logging.info(f'    - {class_name}: Precision={metrics["precision"]:.3f}, '
                         f'Recall={metrics["recall"]:.3f}, F1={metrics["f1"]:.3f}, '
                         f'Support={metrics["support"]}')

    # 최종 confusion matrices 저장
    task_preds, task_true = val_metrics["predictions"]
    plot_confusion_matrices(task_preds, task_true, task_names, f'final_confusion_matrices_{TIMESTAMP}.png')

    # 클래스별 분포 출력
    logging.info("\nPer-task class distribution (Predicted):")
    for i, name in enumerate(task_names):
        preds_array = np.array(task_preds[i])
        unique, counts = np.unique(preds_array, return_counts=True)
        count_dict = dict(zip(unique, counts))
        logging.info(
            f'  - {name}: Negative={count_dict.get(-1, 0)}, Neutral={count_dict.get(0, 0)}, Positive={count_dict.get(1, 0)}')

    # 추가 분석 및 요약
    logging.info(f'\n{"=" * 50}')
    logging.info("FINAL SUMMARY")
    logging.info(f'{"=" * 50}')

    # 가장 성능이 좋은/나쁜 AI 찾기
    best_ai_idx = np.argmax(val_metrics["task_f1_scores"])
    worst_ai_idx = np.argmin(val_metrics["task_f1_scores"])

    logging.info(
        f"Best performing AI: {task_names[best_ai_idx]} (F1: {val_metrics['task_f1_scores'][best_ai_idx]:.4f})")
    logging.info(
        f"Worst performing AI: {task_names[worst_ai_idx]} (F1: {val_metrics['task_f1_scores'][worst_ai_idx]:.4f})")

    # 전체 성능 요약
    overall_accuracy = np.mean(val_metrics["accuracies"])
    overall_f1 = np.mean(val_metrics["task_f1_scores"])

    logging.info(f"\nOverall Performance Summary:")
    logging.info(f"  - Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    logging.info(f"  - Overall F1 Score: {overall_f1:.4f}")
    logging.info(f"  - Exact Match Ratio: {val_metrics['emr']:.4f} ({val_metrics['emr'] * 100:.2f}%)")

    # 각 감성별 평균 성능 분석
    logging.info(f"\nAverage performance per sentiment across all AIs:")
    sentiment_names = ['Negative', 'Neutral', 'Positive']
    sentiment_avg_metrics = {sent: {'precision': [], 'recall': [], 'f1': []} for sent in sentiment_names}

    for task_name in task_names:
        for sentiment in sentiment_names:
            metrics = val_metrics["per_class_metrics"][task_name][sentiment]
            sentiment_avg_metrics[sentiment]['precision'].append(metrics['precision'])
            sentiment_avg_metrics[sentiment]['recall'].append(metrics['recall'])
            sentiment_avg_metrics[sentiment]['f1'].append(metrics['f1'])

    for sentiment in sentiment_names:
        avg_precision = np.mean(sentiment_avg_metrics[sentiment]['precision'])
        avg_recall = np.mean(sentiment_avg_metrics[sentiment]['recall'])
        avg_f1 = np.mean(sentiment_avg_metrics[sentiment]['f1'])
        logging.info(f"  - {sentiment}: Precision={avg_precision:.3f}, Recall={avg_recall:.3f}, F1={avg_f1:.3f}")

    # 가장 어려운 감성 찾기
    sentiment_f1s = [np.mean(sentiment_avg_metrics[sent]['f1']) for sent in sentiment_names]
    hardest_sentiment_idx = np.argmin(sentiment_f1s)
    easiest_sentiment_idx = np.argmax(sentiment_f1s)

    logging.info(
        f"\nEasiest sentiment to predict: {sentiment_names[easiest_sentiment_idx]} (F1: {sentiment_f1s[easiest_sentiment_idx]:.3f})")
    logging.info(
        f"Hardest sentiment to predict: {sentiment_names[hardest_sentiment_idx]} (F1: {sentiment_f1s[hardest_sentiment_idx]:.3f})")

    logging.info(f"\nModel and logs saved. Training session completed at {datetime.now()}")