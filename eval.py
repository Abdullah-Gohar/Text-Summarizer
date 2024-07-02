# Evaluation function
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch
from rouge_score import rouge_scorer
from sum_dataset import SummarizationDataset

data = pd.read_excel('sampled_data.xlsx')
texts = data['Text'].tolist()
summaries = data['Summariez'].tolist()

model_dir = 'Models'
tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def evaluate_model(model, dataset, tokenizer, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    num_samples = len(dataset)

    for i in range(num_samples):
        input_text = dataset.texts[i]
        reference_summary = dataset.summaries[i]
        
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        scores = scorer.score(reference_summary, generated_summary)
        total_scores['rouge1'] += scores['rouge1'].fmeasure
        total_scores['rouge2'] += scores['rouge2'].fmeasure
        total_scores['rougeL'] += scores['rougeL'].fmeasure

    for key in total_scores:
        total_scores[key] /= num_samples

    return total_scores

eval_dataset = SummarizationDataset(texts, summaries, tokenizer)
scores = evaluate_model(model, eval_dataset, tokenizer, device)

print("Evaluation scores:")
print(f"ROUGE-1: {scores['rouge1']}")
print(f"ROUGE-2: {scores['rouge2']}")
print(f"ROUGE-L: {scores['rougeL']}")