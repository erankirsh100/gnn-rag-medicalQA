import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from sentence_transformers import SentenceTransformer
# import faiss
import numpy as np
# from utils import encode_text_and_match_diseases, update_graph_data
from vector_db_files.searcher_class import MilvusSearcher
from gnn_files.gnn import DiseaseSymptomGraphGNN
import pandas as pd
from llm_files.llm_model import run_generation
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
############# Create graph model instance #############
dataset_path = "all_disease_graphs.pkl"
disease_symptom_df_path = "data/disease_csv_files/diseases_symptoms_merged.csv"

dict_paths = {"main_graph_path" : "data/disease_csv_files/diseases_symptoms_merged.csv",
"disease_symptom_csv_path" :  "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv",
"patient_doctor_csv_path" : "data/patient-doctor.csv",
"disease_list_path" : "data/disease_csv_files/unique_aliases.csv",
"disease_aliases_path" : "data/disease_aliases.json",
"symptom_list_path" : "data/disease_csv_files/unique_symptoms.csv"}

df = pd.read_csv(disease_symptom_df_path)
graph_model = DiseaseSymptomGraphGNN(
    df=df,
    dict_paths=dict_paths,
    hidden_channels=128,
    out_channels=1,
    num_layers=2,
    dropout=0.5,
    seed=42,
)

print("creating graph model instance")
graph_model.load_state_dict(torch.load("gnn_files/best_model.pt", map_location=torch.device('cpu')))
collection_name = "pmc_trec_2016"

searcher = None  # global searcher instance


def _init_searcher():
    print("[pipeline] _init_searcher called")
    global searcher
    if searcher is not None:
        print("[pipeline] searcher already initialized", searcher)
        return searcher

    if ':' in os.getenv("PATH_TO_MILVUS_DB", "") and not os.getenv("PATH_TO_MILVUS_DB", "").endswith(".db"):
        print("[pipeline] detected host:port format for Milvus URI")
        uri = os.getenv("PATH_TO_MILVUS_DB", "")
    else:
        print("[pipeline] detected file path format for Milvus URI")
        uri = "./" + os.getenv("PATH_TO_MILVUS_DB", "").split('/')[-1]
    print(f"[pipeline] PATH_TO_MILVUS_DB='{uri}'")
    try:
        searcher = MilvusSearcher(uri=uri, collection_name=collection_name)
        print("[pipeline] MilvusSearcher initialized")
    except Exception as e:
        print("[pipeline] Milvus connection failed at init:", repr(e))
        searcher = None
    return searcher

def run_pipeline(query, reference=None, testing=False, searcher_instance=None):
    print("in run_pipeline with query:", query)
    # ensure searcher available (lazy init)
    global searcher
    max_tries = 3
    while searcher is None and max_tries > 0:
        _init_searcher()
        if searcher is not None:
            break
        print("[pipeline] Retrying in 5 seconds...")
        time.sleep(5)
        max_tries -= 1

    #  get potential diseases from GNN
    graph_results = graph_model.text_forward(query)

    # enrich query with potential diseases for RAG search
    prompt = f"potential diseases: {' ,'.join(graph_results)} \n query: {query}"

    # preform RAG search
    search_results_with_gnn = searcher.search(prompt, limit=5)
    search_results_without_gnn = None
    if testing:
        search_results_without_gnn = searcher.search(f"query: {query}")
    
    # generate final answer
    final_answer = run_generation(query, graph_results, search_results_with_gnn, testing=testing, reference_diagnosis=reference, retrieved_contexts_no_gnn=search_results_without_gnn)
    
    return final_answer

if __name__ == "__main__":
    query = " Hi, my name is XXXX I m a 19year old girl and I keep getting these weird movement feelings in the centre of my stomach (inside) it feel like there is something in there. Sometimes it give me a sharp pain doesn t really hurt just a quick weird pain. At first I thought I could be pregnant but then I tooktook a pregnancy test and it came up negative, and I am also using contraception ( the implant ) so I don t think I could be pregnant, and also the last time I had sex was 5 months ago I feel like I d no if I was pregnant 5 months gone. I just want to know what it is because it is a weird and slightly uncomfortable feeling because I don t know what it is or could be. Thankyou."
    results = run_pipeline(query)
    print("Generated Answer:", results)
