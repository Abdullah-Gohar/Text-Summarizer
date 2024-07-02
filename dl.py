import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch
import time
data = pd.read_excel('sampled_data.xlsx')
texts = data['Text'].tolist()
summaries = data['Summariez'].tolist()

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        targets = self.tokenizer(
            summary, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

tokenizer = BertTokenizer.from_pretrained('facebook/bert-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

train_dataset = SummarizationDataset(texts, summaries, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=0.00005)
num_epochs = 3
total_steps = len(train_loader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=total_steps
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.train()
for epoch in range(num_epochs):
    print(len(train_loader))
    i = 1
    for batch in train_loader:
        print(i)
        start = time.time()
        i += 1
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        print(f"input_ids.shape: {input_ids.shape}")
        print(f"attention_mask.shape: {attention_mask.shape}")
        print(f"labels.shape: {labels.shape}")
        
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        
        
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        end = time.time()
        print("Time Taken: ",end-start)

model.save_pretrained('Models/')
tokenizer.save_pretrained('Models/')
