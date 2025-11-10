import os
import json
import time
import pandas as pd
from multiprocessing import Pool, current_process
from dotenv import load_dotenv

from vector_db_files.searcher_class import MilvusSearcher
from gnn_files.gnn import DiseaseSymptomGraphGNN
from llm_files.llm_model import run_generation_mp

load_dotenv()

# --- Globals shared by each worker ---
graph_models = {}
searchers = {}

def worker_init(process_idx, df, dict_paths, device):
    """Initialize per-process resources once."""
    global graph_models, searchers

    # Each process initializes its own GNN model
    graph_models[process_idx] = DiseaseSymptomGraphGNN(
        df=df,
        dict_paths=dict_paths,
        hidden_channels=128,
        out_channels=1,
        num_layers=2,
        dropout=0.5,
        seed=42,
    )

    # Each process creates one MilvusSearcher connection
    searchers[process_idx] = MilvusSearcher(
        uri=os.getenv("PATH_TO_MILVUS_DB"),
        collection_name="pmc_trec_2016"
    )

def get_results_mp(args):
    row_idx, row_tuple, model_idx = args
    try:
        query = row_tuple.Question
        reference = row_tuple.Answer

        graph_model = graph_models[model_idx]
        searcher = searchers[model_idx]

        # GNN inference
        graph_results = graph_model.text_forward(query)
        prompt = f"potential diseases: {', '.join(graph_results)} \n query: {query}"

        # Retrieve contexts
        search_results_without_gnn = searcher.search(f"query: {query}")
        search_results_with_gnn = searcher.search(prompt, limit=5)

        # LLM generation
        final_answer = run_generation_mp(
            query, graph_results, search_results_with_gnn,
            testing=True, reference_diagnosis=reference,
            retrieved_contexts_no_gnn=search_results_without_gnn,
            api_ex_index=model_idx
        )

        return {
            "Question": query,
            "True_Answer": reference,
            "Basic_llm": final_answer["baseline_results"],
            "Rag_GNN": final_answer["rag_gnn_results"],
            "Rag_only": final_answer["baseline_rag_results"]
        }

    except Exception as e:
        print(f"[ERROR] Worker {model_idx}: {e}")
        return {
            "Question": getattr(row_tuple, 'Question', None),
            "True_Answer": getattr(row_tuple, 'Answer', None),
            "Basic_llm": None,
            "Rag_GNN": None,
            "Rag_only": None
        }

if __name__ == '__main__':
    df = pd.read_csv("data/disease_csv_files/diseases_symptoms_merged.csv")
    icliniq_df_test = pd.read_csv("data/disease_csv_files/test_iclinque_df.csv")

    dict_paths = {
        "main_graph_path": "data/disease_csv_files/diseases_symptoms_merged.csv",
        "disease_symptom_csv_path": "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv",
        "patient_doctor_csv_path": "data/patient-doctor.csv",
        "disease_list_path": "data/disease_csv_files/unique_aliases.csv",
        "disease_aliases_path": "data/disease_aliases.json",
        "symptom_list_path": "data/disease_csv_files/unique_symptoms.csv"
    }

    api_keys = json.loads(os.getenv('API_KEYS') or '[]')
    n = len(api_keys) if api_keys else 1

    print(f"⚙️ Using {n} worker processes...")

    # Init worker pool
    with Pool(
        processes=n,
        initializer=worker_init,
        initargs=(0, df, dict_paths, 'cpu')
    ) as pool:
        limit = 8
        pool_tasks = [(i, row, i % n) for i, row in enumerate(icliniq_df_test.itertuples(index=False), start=0)]

        start_time = time.time()
        results = pool.map(get_results_mp, pool_tasks)
        end_time = time.time()

    res_df = pd.DataFrame(results)
    print(f"✅ Processing time: {end_time - start_time:.2f} seconds")
    res_df.to_csv("results_fast.csv", index=False)
