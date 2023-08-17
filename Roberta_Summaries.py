# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:13:53 2023

@author: karun
"""

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd



# Load your dataset and preprocess it
csv_file_path = "C:\\Users\\karun\\Desktop\\Github\\Kaggle\\Evaluate Student Summaries\\Raw Data\\summaries_train.csv"
df = pd.read_csv(csv_file_path)


# Extracting columns from the DataFrame and storing them in lists
summaries = df['text'].tolist()  # Replace 'summaries' with the actual column name
word_ratings = df['wording'].tolist()  # Replace 'word_ratings' with the actual column name
content_ratings = df['content'].tolist()  # Replace 'content_ratings' with the actual column name


### Create data with relavant columns
data = df[['text', 'content', 'wording']].copy()


# Split data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Load the RoBERTa tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name)
model = RobertaForSequenceClassification(config)

# Prepare data for the model
def preprocess_data(texts, ratings):
    inputs = tokenizer(summaries, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(ratings, dtype=torch.float32)
    return inputs, labels

train_inputs, train_labels = preprocess_data(train_data["text"].tolist(), train_data["wording"].tolist())
val_inputs, val_labels = preprocess_data(val_data["text"].tolist(), val_data["wording"].tolist())

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.MSELoss()

# Fine-tuning loop
num_epochs = 5
batch_size = 16


for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_inputs["input_ids"]), batch_size):
        batch_inputs = {key: val[i:i+batch_size] for key, val in train_inputs.items()}
        batch_labels = train_labels[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(**batch_inputs)
        loss = loss_fn(outputs.logits.squeeze().mean(axis = 1), batch_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(**val_inputs)
        val_predictions = val_outputs.logits.squeeze().mean(axis = 1)

    val_mse = mean_squared_error(val_labels.cpu().numpy(), val_predictions)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation MSE: {val_mse:.4f}")

# Save the trained model
#model.save_pretrained("roberta_regression_model")
#tokenizer.save_pretrained("roberta_regression_model")

































