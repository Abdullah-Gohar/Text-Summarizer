from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

app = Flask(__name__)

model_load_path = 'Models' 
tokenizer = BartTokenizer.from_pretrained(model_load_path)
model = BartForConditionalGeneration.from_pretrained(model_load_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def generate_summary(text, model, tokenizer, device, max_length=512):
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
    input_ids = inputs['input_ids'].to(device)

    attention_mask = inputs['attention_mask'].to(device)

    summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150, num_beams=2, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    if request.method == 'POST':
        text = request.form['text']
        summary = generate_summary(text, model, tokenizer, device)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)