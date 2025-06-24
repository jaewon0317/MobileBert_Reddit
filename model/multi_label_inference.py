import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import MobileBertTokenizer, MobileBertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import ast
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# 1. Multi-Label MobileBERT Model Class (must be same as training)
class MultiLabelMobileBert(nn.Module):
    def __init__(self, model_name='google/mobilebert-uncased', num_labels=12, dropout_rate=0.1):
        super(MultiLabelMobileBert, self).__init__()
        self.num_labels = num_labels
        self.bert = MobileBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # Deeper classifier
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


# 2. Dataset Class
class InferenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256, num_labels=12):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_labels = num_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else None

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

        item = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if label is not None:
            multilabel = convert_to_multilabel(label)
            item['labels'] = torch.tensor(multilabel, dtype=torch.float)
        else:
            item['labels'] = torch.full((self.num_labels,), -1.0, dtype=torch.float)

        return item


# 3. Label Conversion Function
def convert_to_multilabel(labels_list):
    multilabel = np.zeros(12)
    for i, label in enumerate(labels_list):
        if -1 <= label <= 1:
            class_idx = label + 1
            multilabel[i * 3 + class_idx] = 1
    return multilabel


# 4. Constraint Application Function
def apply_constraints(logits):
    batch_size = logits.size(0)
    predictions = torch.zeros_like(logits, dtype=torch.float)

    for i in range(4):
        start_idx = i * 3
        end_idx = start_idx + 3
        ai_logits = logits[:, start_idx:end_idx]
        max_indices = torch.argmax(ai_logits, dim=1)
        for j in range(batch_size):
            predictions[j, start_idx + max_indices[j]] = 1.0

    return predictions


# 5. Inference Function
def inference(model, data_loader, device):
    model.eval()
    all_texts, all_predictions, all_labels_with_placeholders, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Inference'):
            texts = batch['text']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions_constrained = apply_constraints(logits)

            logits_reshaped = logits.view(-1, 4, 3)
            probs = F.softmax(logits_reshaped, dim=2)
            probs_flat = probs.view(-1, 12)

            pred_4d = []
            for pred_12d in predictions_constrained:
                pred_values = []
                for i in range(4):
                    start_idx = i * 3
                    ai_pred_one_hot = pred_12d[start_idx:start_idx + 3]
                    pred_class = torch.argmax(ai_pred_one_hot).item() - 1
                    pred_values.append(pred_class)
                pred_4d.append(pred_values)

            all_texts.extend(texts)
            all_predictions.extend(pred_4d)
            all_probs.extend(probs_flat.cpu().numpy())
            all_labels_with_placeholders.extend(labels)

    results = {
        'texts': all_texts,
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probs)
    }

    if all_labels_with_placeholders:
        results['labels'] = np.array(all_labels_with_placeholders)

    return results


# 6. Analyze Results Function
def analyze_results(results, save_path='inference_results'):
    predictions_4d = results['predictions']
    task_names = ['ChatGPT', 'Claude', 'Grok', 'Gemini']
    os.makedirs(save_path, exist_ok=True)
    print("\n=== Inference Results Analysis ===")

    print("\n1. Prediction Distribution:")
    for i, task_name in enumerate(task_names):
        preds = predictions_4d[:, i]
        unique, counts = np.unique(preds, return_counts=True)
        total = len(preds)
        print(f"\n{task_name}:")
        for val, count in zip(unique, counts):
            sentiment = ['Negative', 'Neutral', 'Positive'][val + 1]
            print(f"  - {sentiment}: {count} ({count / total * 100:.1f}%)")

    if 'labels' in results:
        labels_12d_full = results['labels']
        valid_mask = np.sum(labels_12d_full, axis=1) >= 0
        labels_12d_eval = labels_12d_full[valid_mask]
        predictions_4d_eval = predictions_4d[valid_mask]

        if len(labels_12d_eval) == 0:
            print("\n2. Performance Metrics: No labeled data found to evaluate.")
        else:
            labels_4d_eval = []
            for label_12 in labels_12d_eval:
                label_4 = [np.argmax(label_12[i * 3:i * 3 + 3]) - 1 for i in range(4)]
                labels_4d_eval.append(label_4)
            labels_4d_eval = np.array(labels_4d_eval)

            print(f"\n2. Performance Metrics (evaluated on {len(labels_4d_eval)} labeled samples):")
            accuracies = []
            for i, task_name in enumerate(task_names):
                acc = np.mean(predictions_4d_eval[:, i] == labels_4d_eval[:, i])
                accuracies.append(acc)
                print(f"\n{task_name} Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
            print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} ({np.mean(accuracies) * 100:.2f}%)")
            emr = np.mean(np.all(predictions_4d_eval == labels_4d_eval, axis=1))
            print(f"Exact Match Ratio (EMR): {emr:.4f} ({emr * 100:.2f}%)")

            print("\n3. Detailed Classification Reports:")
            for i, task_name in enumerate(task_names):
                print(f"\n{task_name}:")
                print(classification_report(labels_4d_eval[:, i], predictions_4d_eval[:, i],
                                            target_names=['Negative', 'Neutral', 'Positive'], digits=4))

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            for i, task_name in enumerate(task_names):
                cm = confusion_matrix(labels_4d_eval[:, i], predictions_4d_eval[:, i], labels=[-1, 0, 1])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                            xticklabels=['Negative', 'Neutral', 'Positive'],
                            yticklabels=['Negative', 'Neutral', 'Positive'])
                axes[i].set_title(f'{task_name} Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('True')
            plt.tight_layout()
            plt.savefig(f'{save_path}/confusion_matrices.png')
            plt.close()
            print(f"\nConfusion matrices saved to {save_path}/confusion_matrices.png")


# 7. Save Results Function (수정됨)
def save_results(results, df_original, output_path='inference_output.csv'):
    predictions = results['predictions']
    probs = results['probabilities']
    df_results = df_original.copy()

    df_results['pred_vector'] = [pred.tolist() for pred in predictions]

    # prob_vector를 순수 파이썬 소수 리스트로 변환하고 반올림
    # NumPy 배열 자체의 .round().tolist() 메서드를 사용하여 타입 변환 및 반올림
    # 요청하신대로 소수점 3자리까지 반올림합니다.
    df_results['prob_vector'] = [prob.round(3).tolist() for prob in probs]

    if 'labels_list' in df_results.columns:
        df_results = df_results.drop(columns=['labels_list'])

    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {output_path}")
    return df_results


# 8. Main Execution Block
if __name__ == "__main__":
    MODEL_PATH = "multilabel_mobilebert_best_20250624_105658.pt"
    DATA_PATH = "inference_output_20250624_132407.csv"
    MODEL_NAME = 'google/mobilebert-uncased'
    MAX_LEN, BATCH_SIZE, DROPOUT_RATE, NUM_LABELS = 256, 32, 0.1, 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    df['Data'] = df['Data'].fillna('')

    label_column = 'label(chatgpt,claude,grok,gemini)'
    if label_column in df.columns:
        has_label = df[label_column].notna()
        df.loc[has_label, 'labels_list'] = df.loc[has_label, label_column].apply(ast.literal_eval)
        print(f"Loaded {len(df)} samples total:")
        print(f"  - With labels: {has_label.sum()}")
        print(f"  - Without labels: {(~has_label).sum()}")
        labels = [row['labels_list'] if has_label.loc[idx] else None for idx, row in df.iterrows()]
        labels = np.array(labels, dtype=object)
    else:
        labels = None
        print(f"Loaded {len(df)} samples without a label column")
    texts = df['Data'].values

    print("\nLoading tokenizer...")
    tokenizer = MobileBertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    dataset = InferenceDataset(texts, labels, tokenizer, max_len=MAX_LEN, num_labels=NUM_LABELS)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nLoading model from {MODEL_PATH}...")
    model = MultiLabelMobileBert(model_name=MODEL_NAME, num_labels=NUM_LABELS, dropout_rate=DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")

    print("\nPerforming inference...")
    results = inference(model, data_loader, device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'multilabel_inference_results_{timestamp}'
    analyze_results(results, save_path)

    output_file = f'multilabel_inference_output_{timestamp}.csv'
    df_results = save_results(results, df, output_file)

    print("\n=== Sample Predictions ===")
    task_names = ['ChatGPT', 'Claude', 'Grok', 'Gemini']
    for i in range(min(5, len(df))):
        print(f"\nSample {i + 1}:")
        print(f"Text: {df['Data'].iloc[i][:100]}...")
        print(f"Prediction Vector: {df_results['pred_vector'].iloc[i]}")

        prob_vec = df_results['prob_vector'].iloc[i]
        print("Probability Vector:")
        for j, task_name in enumerate(task_names):
            probs_task = prob_vec[j * 3: j * 3 + 3]
            print(f"  - {task_name}: [neg={probs_task[0]:.3f}, neu={probs_task[1]:.3f}, pos={probs_task[2]:.3f}]")

        if 'labels_list' in df.columns and isinstance(df['labels_list'].iloc[i], list):
            print(f"True labels: {df['labels_list'].iloc[i]}")
        else:
            print("True labels: Not available (to be predicted)")

    print("\nInference completed!")