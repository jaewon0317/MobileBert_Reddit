import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device", device)

data_path = "reviews.csv"
df = pd.read_csv(data_path,encoding="cp949")
data_X = list(df['Text'].values)
labels = df['Sentiment'].values
print(len(data_X))

model = MobileBertForSequenceClassification.from_pretrained("mobliebert_custom_model_imdb.pt")
model.to(device)
model.eval()

tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased",do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding='max_length')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print('토큰화 완료')

batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = torch.utils.data.RandomSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler,batch_size=batch_size)
print('데이터셋 구축 완료')



test_pred = []
test_true = []
for batch in tqdm(test_dataloader, desc ='Inferencing Full Dataset'):
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
    (''
     '')

test_accuracy = np.sum( np.array(test_pred) == np.array(test_true) ) / len(test_pred)
print('전체 데이터 49,998 건에 대한 영화리뷰 긍부정 정확도:',test_accuracy)