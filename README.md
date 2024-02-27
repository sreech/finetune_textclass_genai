# Fine-Tuning for Movie Magic: Classifying Reviews with Hugging Face AutoTrain:
Welcome, movie buffs and data enthusiasts! Today, I'm excited to share my journey into the world of fine-tuning with Hugging Face AutoTrain. I've built a text classification model specifically designed to analyze movie reviews and predict their sentiment – positive or negative. Buckle up, because we're about to dive into the world of sentiment analysis and explore the power of fine-tuning pre-trained models!
 I utilized a dataset of 200 movie reviews, each paired with its corresponding sentiment label (positive or negative).
 I chose DistilBERT/DistilRoBERTa-base as my pre-trained model of choice. This mighty model, boasting 82.8 million parameters, has already been trained on a massive dataset of text and code, giving it a strong foundation for understanding language nuances.

# Code to use in your colab or notebook
```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("sreerammadhu/sreegeni-finetune-textclass-auto")
model = AutoModelForSequenceClassification.from_pretrained("sreerammadhu/sreegeni-finetune-textclass-auto")

text = "This movie was fantastic!"  # Replace with your text
encoded_text = tokenizer(text, return_tensors="pt")
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**encoded_text)
    logits = outputs.logits  # Get the logits (model's predictions)
```
# Extract predicted label and confidence score
```
predicted_label = logits.argmax(-1).item()
confidence_score = logits.softmax(-1)[0][predicted_label].item()
```
# Map predicted label back to its meaning (assuming labels are 0 for negative, 1 for positive)
```
labels = ["Negative", "Positive"]
predicted_sentiment = labels[predicted_label]

print(f"Predicted sentiment: {predicted_sentiment} with confidence score: {confidence_score:.4f}")
```
---
tags:
- autotrain
- text-classification
widget:
- text: I love AutoTrain
datasets:
- sreegeni-finetune-textclass-auto/autotrain-data
license: apache-2.0
pipeline_tag: text-classification
---

# Model Trained Using AutoTrain

- Problem type: Text Classification

# Implement using Python autotrain command in Colab
```
import os
!pip install -U autotrain-advanced >install_logs.txt
!autotrain setup --colab > setup_logs.txt

!pip install datasets >install_logs.txt
from datasets import load_dataset
# Load the dataset
dataset = load_dataset("tatsu-lab/alpaca")
dataset
train = dataset['train']
train
train.to_csv('train.csv', index = False)
```
# Place the train.csv generated above in the data folder of colab
```
import os
project_name = 'sreegenai-finetune-textclass-llm' # @param {type:"string"}
model_name = 'distilbert/distilroberta-base' # @param {type:"string"}
data_path = "data/" # @param {type:"string"}
push_to_hub = True # @param ["False", "True"] {type:"raw"}
hf_token = "hf_PUTYOURS" #@param {type:"string"}
repo_id = "sreerammadhu/sreegen-finetune-python" #@param {type:"string"}
learning_rate = 2e-4 # @param {type:"number"}
num_epochs = 1 #@param {type:"number"}
batch_size = 1 # @param {type:"slider", min:1, max:32, step:1}
block_size = 1024 # @param {type:"number"}
warmup_ratio = 0.1 # @param {type:"number"}
weight_decay = 0.01 # @param {type:"number"}
gradient_accumulation = 4 # @param {type:"number"}
peft = True # @param ["False", "True"] {type:"raw"}
quantization = "int4" # @param ["int4", "int8", "none"] {type:"raw"}
lora_r = 16 #@param {type:"number"}
lora_alpha = 32 #@param {type:"number"}
lora_dropout = 0.05 #@param {type:"number"}

os.environ["PROJECT_NAME"] = project_name
os.environ["MODEL_NAME"] = model_name
os.environ["DATA_PATH"] = data_path
os.environ["PUSH_TO_HUB"] = str(push_to_hub)
os.environ["HF_TOKEN"] = hf_token
os.environ["REPO_ID"] = repo_id
os.environ["LEARNING_RATE"] = str(learning_rate)
os.environ["NUM_EPOCHS"] = str(num_epochs)
os.environ["BATCH_SIZE"] = str(batch_size)
os.environ["BLOCK_SIZE"] = str(block_size)
os.environ["WARMUP_RATIO"] = str(warmup_ratio)
os.environ["WEIGHT_DECAY"] = str(weight_decay)
os.environ["GRADIENT_ACCUMULATION"] = str(gradient_accumulation)
os.environ["PEFT"] = str(peft)
os.environ["QUANTIZATION"] = str(quantization)
os.environ["LORA_R"] = str(lora_r)
os.environ["LORA_ALPHA"] = str(lora_alpha)
os.environ["LORA_DROPOUT"] = str(lora_dropout)


!autotrain llm \
--train \
--model ${MODEL_NAME} \
--project-name ${PROJECT_NAME} \
--data-path data/ \
--text-column review \
--lr ${LEARNING_RATE} \
--batch-size ${BATCH_SIZE} \
--epochs ${NUM_EPOCHS} \
--block-size ${BLOCK_SIZE} \
--warmup-ratio ${WARMUP_RATIO} \
--lora-r ${LORA_R} \
--lora-alpha ${LORA_ALPHA} \
--lora-dropout ${LORA_DROPOUT} \
--weight-decay ${WEIGHT_DECAY} \
--gradient-accumulation ${GRADIENT_ACCUMULATION} \
--quantization ${QUANTIZATION} \
--merge-adapter \
--push-to-hub \
--token ${HF_TOKEN} \
--repo-id ${REPO_ID}
```

## Validation Metrics
loss: 0.4902153015136719

f1: 0.7368421052631577

precision: 0.8235294117647058

recall: 0.6666666666666666

auc: 0.899749373433584

accuracy: 0.75
