import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
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


# 1. Multi-Head MobileBERT Model Class (must be same as training)
class MultiHeadMobileBert(nn.Module):
    def __init__(self, model_name='google/mobilebert-uncased', num_tasks=4, num_classes_per_task=3, dropout_rate=0.1):
        super(MultiHeadMobileBert, self).__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.bert = MobileBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # Deeper classifier
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


# 2. Dataset Class (Corrected)
class InferenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256, num_tasks=4):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_tasks = num_tasks

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
            item['labels'] = torch.tensor(label, dtype=torch.long) + 1
        else:
            item['labels'] = torch.full((self.num_tasks,), -100, dtype=torch.long)

        return item


# 3. Inference Function (Corrected)
def inference(model, data_loader, device, task_names):
    model.eval()
    all_texts, all_predictions, all_labels_with_placeholders, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Inference'):
            texts = batch['text']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            logits_list = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds, batch_probs = [], []

            for logits in logits_list:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                batch_preds.append(preds.cpu().numpy())
                batch_probs.append(probs.cpu().numpy())

            batch_preds = np.array(batch_preds).T
            batch_probs = np.array(batch_probs).transpose(1, 0, 2)

            all_texts.extend(texts)
            all_predictions.extend(batch_preds - 1)
            all_probs.extend(batch_probs)
            all_labels_with_placeholders.extend(labels)

    labels_full = np.array(all_labels_with_placeholders)
    has_any_labels = np.any(labels_full != -100)

    results = {
        'texts': all_texts,
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probs),
        'task_names': task_names
    }

    if has_any_labels:
        results['labels'] = labels_full - 1

    return results


# 4. Analyze Results Function
def analyze_results(results, save_path='inference_results'):
    predictions = results['predictions']
    task_names = results['task_names']
    os.makedirs(save_path, exist_ok=True)

    print("\n=== Inference Results Analysis ===")
    print("\n1. Prediction Distribution:")
    for i, task_name in enumerate(task_names):
        preds = predictions[:, i]
        unique, counts = np.unique(preds, return_counts=True)
        total = len(preds)
        print(f"\n{task_name}:")
        for val, count in zip(unique, counts):
            sentiment = ['Negative', 'Neutral', 'Positive'][val + 1]
            print(f"  - {sentiment}: {count} ({count / total * 100:.1f}%)")

    if 'labels' in results:
        labels_full = results['labels']
        valid_mask = np.all(labels_full != -101, axis=1)
        labels_eval = labels_full[valid_mask]
        predictions_eval = predictions[valid_mask]

        if len(labels_eval) == 0:
            print("\n2. Performance Metrics: No labeled data found to evaluate.")
        else:
            print(f"\n2. Performance Metrics (evaluated on {len(labels_eval)} labeled samples):")
            accuracies = []
            for i, task_name in enumerate(task_names):
                acc = np.mean(predictions_eval[:, i] == labels_eval[:, i])
                accuracies.append(acc)
                print(f"\n{task_name} Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
            print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} ({np.mean(accuracies) * 100:.2f}%)")
            emr = np.mean(np.all(predictions_eval == labels_eval, axis=1))
            print(f"Exact Match Ratio (EMR): {emr:.4f} ({emr * 100:.2f}%)")

            print("\n3. Detailed Classification Reports:")
            for i, task_name in enumerate(task_names):
                print(f"\n{task_name}:")
                print(classification_report(labels_eval[:, i], predictions_eval[:, i],
                                            target_names=['Negative', 'Neutral', 'Positive'],
                                            digits=4))

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            for i, task_name in enumerate(task_names):
                cm = confusion_matrix(labels_eval[:, i], predictions_eval[:, i], labels=[-1, 0, 1])
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

    print("\n4. Prediction Confidence Analysis:")
    probs = results['probabilities']
    for i, task_name in enumerate(task_names):
        max_probs = np.max(probs[:, i, :], axis=1)
        avg_confidence = np.mean(max_probs)
        print(f"\n{task_name}:")
        print(f"  - Average Confidence: {avg_confidence:.4f}")
        print(f"  - Min Confidence: {np.min(max_probs):.4f}")
        print(f"  - Max Confidence: {np.max(max_probs):.4f}")
        low_conf_ratio = np.mean(max_probs < 0.5)
        print(f"  - Low Confidence Predictions (<0.5): {low_conf_ratio * 100:.1f}%")


# 5. Save Results Function (수정됨)
def save_results(results, df_original, output_path='inference_output.csv'):
    predictions = results['predictions']
    probs = results['probabilities']
    task_names = results['task_names']
    df_results = df_original.copy()

    df_results['pred_vector'] = [pred.tolist() for pred in predictions]

    # prob_vector를 순수 파이썬 소수 리스트로 변환하고 반올림
    prob_vectors = []
    for i in range(len(predictions)):
        prob_vector = []
        for j in range(len(task_names)):
            # NumPy 배열 자체의 .round().tolist() 메서드를 사용하여 타입 변환 및 반올림
            # 요청하신대로 소수점 3자리까지 반올림합니다.
            rounded_probs = probs[i, j, :].round(3).tolist()
            prob_vector.extend(rounded_probs)
        prob_vectors.append(prob_vector)
    df_results['prob_vector'] = prob_vectors

    if 'labels_list' in df_results.columns:
        df_results = df_results.drop(columns=['labels_list'])

    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {output_path}")
    return df_results


# 6. Main Execution Block
if __name__ == "__main__":
    MODEL_PATH = "multihead_mobilebert_best_20250624_111646.pt"
    DATA_PATH = "labeled_data.csv"
    MODEL_NAME = 'google/mobilebert-uncased'
    MAX_LEN, BATCH_SIZE, DROPOUT_RATE = 256, 32, 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    df['Data'] = df['Data'].fillna('')

    label_column = 'label(chatgpt,claude,grok,gemini)'
    task_names = ['ChatGPT', 'Claude', 'Grok', 'Gemini']

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

    dataset = InferenceDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        num_tasks=len(task_names)
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nLoading model from {MODEL_PATH}...")
    model = MultiHeadMobileBert(
        model_name=MODEL_NAME,
        num_tasks=len(task_names),
        num_classes_per_task=3,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")

    print("\nPerforming inference...")
    results = inference(model, data_loader, device, task_names)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'inference_results_{timestamp}'
    analyze_results(results, save_path)

    output_file = f'inference_output_{timestamp}.csv'
    df_results = save_results(results, df, output_file)

    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(df))):
        print(f"\nSample {i + 1}:")
        print(f"Text: {df['Data'].iloc[i][:100]}...")
        print(f"Prediction Vector: {df_results['pred_vector'].iloc[i]}")

        prob_vec = df_results['prob_vector'].iloc[i]
        print("Probability Vector:")
        for j, task_name in enumerate(task_names):
            start_idx = j * 3
            probs_task = prob_vec[start_idx:start_idx + 3]
            print(f"  - {task_name}: [neg={probs_task[0]:.3f}, neu={probs_task[1]:.3f}, pos={probs_task[2]:.3f}]")

        if 'labels_list' in df.columns and isinstance(df['labels_list'].iloc[i], list):
            true_labels = df['labels_list'].iloc[i]
            print(f"True labels: {true_labels}")
        else:
            print("True labels: Not available (to be predicted)")

    print("\nInference completed!")