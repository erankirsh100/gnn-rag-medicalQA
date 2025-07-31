from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pandas as pd

# Load sentence-transformer model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
disease_list = pd.read_csv("data/disease_csv_files/unique_aliases.csv")["0"].tolist()
simptoms_list = pd.read_csv("data/disease_csv_files/unique_symptoms.csv")["symptoms"].tolist()

def bert_encode(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use CLS token

def encode_text_and_match_diseases(text, top_k=5):
    encoding_dict = {}
    text_lower = text.lower()
    lists = [disease_list, simptoms_list]
    text_vec = bert_encode(text)
    encoding_dict["text_vector"] = text_vec[0].tolist()

    for idx, lst in enumerate(lists):
        lst = [item.lower() for item in lst]
        # 1. Find exact/partial disease matches in text
        exact_matches = [disease for disease in lst if re.search(rf'\b{re.escape(disease.lower())}\b', text_lower)]
        
        disease_vecs = np.vstack([bert_encode(disease) for disease in lst])
        
        # 3. Compute cosine similarities
        similarities = cosine_similarity(text_vec, disease_vecs)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        semantic_matches = [(disease_list[i], similarities[i]) for i in top_indices]

        encoding_dict[f"exact_matches_{idx}"] = exact_matches
        encoding_dict[f"semantic_matches_{idx}"] = semantic_matches

    return encoding_dict

def update_graph_data(graph_data, new_data):
    pass