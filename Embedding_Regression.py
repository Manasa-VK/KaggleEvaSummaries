# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:18:53 2023

@author: karun
"""

import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


# Load your dataset and preprocess it
csv_file_path = "C:\\Users\\karun\\Desktop\\Github\\Kaggle\\Evaluate Student Summaries\\Raw Data\\summaries_train.csv"
df = pd.read_csv(csv_file_path)


#### model development
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




model = SentenceTransformer('BAAI/bge-large-en')
text_embeddings = model.encode(df['text'].tolist(), normalize_embeddings=True, show_progress_bar=True)

df['text_embeddings'] = text_embeddings












