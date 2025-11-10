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
from llm_files.llm_model import run_generation_mp
from time import time
import os
from dotenv import load_dotenv
import json
from multiprocessing import Pool


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

# Create n locks for the n graph models
n = len(json.loads(os.getenv('API_KEYS'))) if os.getenv('API_KEYS') else 1
# print(f"Using {n} processes for multiprocessing pipeline.")

def get_results_mp(args):
    print("Process started")
    try:
        (row_idx, row), model_idx, graph_model = args

        print("thread model idx: ", model_idx, " started processing query.")
        query = row.Question
        reference = row.Answer
        # Acquire lock for the graph model
        # with n_locks[model_idx]:
        #     graph_model = n_graph_models[model_idx]
        #     graph_results = graph_model.text_forward(query)
        graph_results = graph_model.text_forward(query)
        
        prompt = f"potential diseases: {' ,'.join(graph_results)} \n query: {query}"

        print("thread model idx: ", model_idx, " created prompt for search.")

        # All processes will share the same searcher instance with lock mechanism

        print("creating searcher instance in process with model idx:", model_idx)

        searcher = MilvusSearcher(uri=os.getenv("PATH_TO_MILVUS_DB"), collection_name="pmc_trec_2016")
        
        search_results_without_gnn = searcher.search(f"query: {query}")
        search_results_with_gnn = searcher.search(prompt, limit=5)

        print("thread model idx: ", model_idx, " completed search.")
        
        # run_generation(query, graph_results, search_results_with_gnn, testing=testing, reference_diagnosis=reference, retrieved_contexts_no_gnn=search_results_without_gnn)
        final_answer = run_generation_mp(query, graph_results, search_results_with_gnn, testing=True, reference_diagnosis=reference, retrieved_contexts_no_gnn=search_results_without_gnn, api_ex_index=model_idx)

        print("thread model idx: ", model_idx, " completed LLM generation.")
        return {
            "Question": query,
            "True_Answer": reference,
            "Basic_llm": final_answer["baseline_results"],
            "Rag_GNN": final_answer["rag_gnn_results"],
            "Rag_only": final_answer["baseline_rag_results"]
        }
    except Exception as e:
        print("Error in process with model idx:", model_idx, " Error:", str(e))
        return {
            "Question": query,
            "True_Answer": reference,
            "Basic_llm": None,
            "Rag_GNN": None,
            "Rag_only": None
        }

if __name__ == '__main__':

    df = pd.read_csv(disease_symptom_df_path)
    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We will be using n processes to parallelize the pipeline

    api_keys_str = os.getenv('API_KEYS')
    api_keys = json.loads(api_keys_str) if api_keys_str else []

    print(f"Using {n} processes for multiprocessing pipeline.")

    # Create n separate graph_model for each process
    print(f"Creating {n} graph model instances.")
    n_graph_models = [DiseaseSymptomGraphGNN(
        df=df,
        dict_paths=dict_paths,
        hidden_channels=128,
        out_channels=1,
        num_layers=2,
        dropout=0.5,
        seed=42,
    ) for _ in range(n)]

    # print(f"Creating {n} searchers instances.")
    # n_searchers = [MilvusSearcher(uri=os.getenv("PATH_TO_MILVUS_DB"), collection_name="pmc_trec_2016") for _ in range(n)]

    # load dataset
    icliniq_df_test = pd.read_csv("data/disease_csv_files/test_iclinque_df.csv")
    
    # Running multiprocessing #
    start_time = time()

    limit = 8
    # print("here")
    pool_tasks = [(row, i%n, n_graph_models[i%n]) for i, row in enumerate(icliniq_df_test[:limit].iterrows())]

    # print("pool tasks created:", pool_tasks)
    with Pool(processes=n) as pool:
        # print("Pool created, starting mapping tasks.")
        results = pool.map(get_results_mp, pool_tasks)
        # print("Mapping tasks completed.")

    res_df = pd.DataFrame(results)

    end_time = time()
    print(f"Processing time: {end_time - start_time} seconds")