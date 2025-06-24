import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import batched_powerSGD_hook
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention

# 0.Gpu 있는지 확인, 없는 경우에는  CPU로 구동됨
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device",device)

# 1. 학습 시 경고 메세지 제거
logging.set_verbosity_error()

#2.데이터 확인
path = 'imdb_reviews_sample.csv'
df = pd.read_csv(path, encoding='cp949')
data_X = list(df['Text'].values)
labels = df['Sentiment'].values

print('### 데이터 샘플 ###')
print("리뷰 문장:",data_X[:5])
print("긍정/부정:",labels[:5])

# 3. 텍스트를 토큰으로 나눔(토큰화)
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased',do_lower_case = True)
inputs = tokenizer(data_X, truncation=True,max_length=256, add_special_tokens=True, padding="max_length")
print(type(inputs))
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

num_to_print = 3
print('\n###토큰화 결과 샘플 ###')
for j in range(num_to_print):
    print(f'\n{j+1}번째 데이터')
    print('데이터 :',data_X[j])
    print('토큰:',input_ids[j])
    print('어탠션 마스크:',attention_mask[j])

#4. 학습용 및 검증용 데이터셋 분리(scikil learn 에 있는 train_test_split 함수 사용, random_state 일치시켜야함)
train, validation, train_y,validation_y = train_test_split(input_ids,labels,test_size=0.2,random_state=2025)
train_masks,validation_masks, _,_ = train_test_split(attention_mask,labels,test_size=0.2,random_state=2025)

#5. MobileBert에 영화 리뷰 데이터를 Finetuning 하기 위한 데이터 설정
batch_size = 8
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_masks)
train_data = TensorDataset(train_inputs,train_masks,train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

#검증용 데이터로더 구현
validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_masks)
validation_data = TensorDataset(validation_inputs,validation_masks,validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler,batch_size=batch_size)

# 6. 모델 설정
model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased',num_labels=2)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader)*epochs)

#7. 학습(Loss),검증 (train accuracy, validation accuracy)
epochs_results = []

for e in range(epochs):
    #학습 루프
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader,desc = f"Training Epoch{e+1}",leave=True)
    for batch in progress_bar:
        batch_ids, batch_mask, batch_labels = batch

        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        model.zero_grad()

        #앞먹임 : forward pass
        output = model(batch_ids,attention_mask=batch_mask,labels = batch_labels)
        loss = output.loss
        total_train_loss += loss.item()

        #역전파 : backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    #학습 데이터셋에 대한 평균 손실값 계산
    avg_train_loss = total_train_loss / len(train_dataloader)

    #학습 데디터셋에 대한 정확도(accuracy) 계산
    model.eval()
    train_pred=[]
    train_true=[]

    for batch in tqdm(train_dataloader,desc=f"evaluation Train Epoch{e+1}",leave =True):
        batch_ids, batch_mask, batch_labels =batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids,attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits,dim =1)
        train_pred.extend(pred.cpu().numpy())
        train_true.extend(batch_labels.cpu().numpy())

    train_accuracy = np.sum(np.array(train_pred) == np.array(train_true)) / len(train_pred)

    #검증 데이터셋에 대한 정확도(accuracy) 계산

    val_pred = []
    val_true = []

    for batch in tqdm(validation_dataloader, desc=f"evaluation validation Epoch{e + 1}", leave=True):
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        val_pred.extend(pred.cpu().numpy())
        val_true.extend(batch_labels.cpu().numpy())

    val_accuracy = np.sum(np.array(val_pred) == np.array(val_true)) / len(val_pred)
    epochs_results.append((avg_train_loss,train_accuracy,val_accuracy))

# 8. 학습 종료후 epoch별 학습 경과 및 검증 정확도 출력
for idx, (loss,train_acc,val_acc) in enumerate(epochs_results, start=1):
    print(f"Epoch{idx}: Train loss:{loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy:{val_acc:.4f}")

#9. 모델 저장
print('\n ### 모델 저장 ###')
save_path = 'mobliebert_custom_model_imdb'
model.save_pretrained(save_path + '.pt')
print('모델저장 완료')
