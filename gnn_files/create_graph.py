import torch
import pandas as pd
from torch_geometric.data import Data
import torch
from torch_geometric.data import Data
import pandas as pd
import pickle
import numpy as np

from torch_geometric.nn import SAGEConv
import torch
import torch_geometric
import torch.nn.functional as F
torch_geometric.set_debug(True)
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pandas as pd
from gnn_files.utils import bert_encode
import json
from tqdm import tqdm  # make sure to install with: pip install tqdm


import torch
import pandas as pd
from torch_geometric.data import Data
import torch
from torch_geometric.data import Data
import pandas as pd
import pickle
import numpy as np

from torch_geometric.nn import SAGEConv
import torch
import torch_geometric
import torch.nn.functional as F
torch_geometric.set_debug(True)
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pandas as pd
# from utils import bert_encode
import json
from tqdm import tqdm  # make sure to install with: pip install tqdm

import re
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch_geometric
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # pip install tqdm
from transformers import AutoTokenizer, AutoModel

# from utils import bert_encode  # custom function

torch_geometric.set_debug(True)


class DiseaseKnowledgeGraph:
    def __init__(
        self,
        main_graph_path,
        disease_symptom_csv_path,
        patient_doctor_csv_path,
        disease_list_path,
        symptom_list_path,
        disease_aliases_path=None, 
        qa_data_path=None
        
    ):
        # === Load data ===
        self.main_graph_df = pd.read_csv(main_graph_path)
        self.disease_symptom_csv = pd.read_csv(disease_symptom_csv_path)
        self.patient_doctor_csv = pd.read_csv(patient_doctor_csv_path)
        self.qa_data = pd.read_csv(qa_data_path) if qa_data_path else None

        # Load disease list & mapping
        self.disease_list = sorted(self.main_graph_df["diseases"].dropna().unique().tolist())
        print(f'Number of unique diseases: {len(self.disease_list)}')
        self.disease2id = {disease: i for i, disease in enumerate(self.disease_list)}

        # Load symptoms
        self.symptom_list = pd.read_csv(symptom_list_path)["symptoms"].tolist()
        self.alias_disease_list = pd.read_csv(disease_list_path)["0"].tolist()

        # Load disease aliases
        self.disease2aliases = self.load_disease_aliases(disease_aliases_path)
        if self.disease2aliases:
            self.alias2disease = {
                alias.lower(): disease
                for disease, aliases in self.disease2aliases.items()
                for alias in aliases
            }
        else:
            self.alias2disease = {}

        # === Create main graph ===
        self.main_graph, self.node2id, self.num_node_features = self.create_graph(self.main_graph_df)
        self.graphs = []

        # === Build QA graphs ===
        if self.qa_data is not None:
            for _, row_data in tqdm(
                self.qa_data.sample(n=1500, random_state=42).iterrows(),
                total=1500,
                desc="Building QA graphs"
            ):
                
                text_patient = row_data["Question"]
                data_patient = self.encode_text_and_match_diseases(text_patient)

                text_doctor = row_data["Answer"]
                data_doctor = self.encode_text_and_match_diseases(text_doctor)

                disease = data_doctor["exact_matches_disease"]
                if len(disease) == 0:
                    disease_in_list = data_doctor["semantic_matches_disease"][0:1]
                    disease = [disease_in_list[0][0]]
                self.graphs.append(self.update_graph_data(data_patient, disease))
            
        # # === Build patient-doctor graphs ===
        # for disease in tqdm(self.disease_list, desc="Building disease-symptom graphs"):
        #     # Get all rows for this disease
        #     disease_rows = self.disease_symptom_csv[self.disease_symptom_csv["diseases"] == disease]

        #     # Sample up to 20 random examples (if fewer than 20, take all)
        #     sampled_rows = disease_rows.sample(n=min(4, len(disease_rows)), random_state=42)

        #     for _, row_data in sampled_rows.iterrows():
        #         symptoms = [
        #             self.symptom_list[i]
        #             for i, val in enumerate(row_data[1:].tolist())  # assuming first col is "diseases"
        #             if val == 1
        #         ]

        #         data = self.encode_text_and_match_diseases("person")
        #         data["exact_matches_symptom"].extend(symptoms)
        #         self.graphs.append(self.update_graph_data(data, disease))


        # for _, row_data in tqdm(
        #     self.patient_doctor_csv.sample(n=1500, random_state=42).iterrows(),
        #     total=1500,
        #     desc="Building patient-doctor graphs"
        # ):
        #     text_patient = row_data["problem_description"]
        #     data_patient = self.encode_text_and_match_diseases(text_patient)

        #     text_doctor = row_data["Doctor"]
        #     data_doctor = self.encode_text_and_match_diseases(text_doctor)

        #     disease = data_doctor["exact_matches_disease"]
        #     if len(disease) == 0:
        #         disease_in_list = data_doctor["semantic_matches_disease"][0:1]
        #         disease = [disease_in_list[0][0]]
        #     self.graphs.append(self.update_graph_data(data_patient, disease))

    def load_disease_aliases(self, disease_aliases_path):
        if disease_aliases_path is not None:
            with open(disease_aliases_path, "r") as f:
                return json.load(f)
        return {}

    def encode_text_and_match_diseases(self, text, top_k=5):
        encoding_dict = {}
        text_lower = text.lower()
        lists = [("disease", self.alias_disease_list), ("symptom", self.symptom_list)]
        text_vec = bert_encode(text)
        encoding_dict["text_vector"] = text_vec[0].tolist()

        for list_type, lst in lists:
            lst_lower = [item.lower() for item in lst]

            # Exact matches
            exact_matches = [
                entity for entity in lst_lower
                if re.search(rf"\b{re.escape(entity)}\b", text_lower)
            ]

            # Semantic matches
            entity_vecs = np.vstack([bert_encode(entity) for entity in lst_lower])
            similarities = cosine_similarity(text_vec, entity_vecs)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]
            semantic_matches = [(lst_lower[i], similarities[i]) for i in top_indices]

            encoding_dict[f"exact_matches_{list_type}"] = exact_matches
            encoding_dict[f"semantic_matches_{list_type}"] = semantic_matches

        return encoding_dict

    def create_graph(self, df):
        person_node = "person"
        disease_nodes = df["diseases"].unique().tolist()
        symptom_nodes = df.columns[1:].tolist()

        all_nodes = [person_node] + disease_nodes + symptom_nodes
        node2id = {name: i for i, name in enumerate(all_nodes)}

        mask = np.zeros((len(all_nodes),), dtype=bool)
        for node in disease_nodes:
            mask[node2id[node]] = 1
        
        edges, edge_types = [], []

        # person → disease (type 0)
        for disease in disease_nodes:
            edges.append([node2id[person_node], node2id[disease]])
            edge_types.append(0)

        # disease → symptom (type 1)
        for _, row in df.iterrows():
            disease = row["diseases"]
            for symptom in symptom_nodes:
                if row[symptom] == 1:
                    edges.append([node2id[disease], node2id[symptom]])
                    edge_types.append(1)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long)

        # === Node features ===
        node_features = []
        for node in all_nodes:
            vec = bert_encode(node)
            if isinstance(vec, (np.ndarray, torch.Tensor)):
                vec = vec.flatten().tolist()
            else:
                vec = list(vec)

            if node == person_node:
                node_features.append([0, 1] + vec)
            elif node in disease_nodes:
                node_features.append([1, 0] + vec)
            else:
                node_features.append([2, 0] + vec)

        x = torch.tensor(node_features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        data.disease_mask = torch.tensor(mask, dtype=torch.bool)
        return data, node2id, x.shape[1]

    def update_graph_data(self, new_data, list_y):
        gnn_graph = self.main_graph.clone()
        disease_list = (
            new_data.get("exact_matches_disease", [])
            + [d for d, _ in new_data.get("semantic_matches_disease", [])]
        )
        symptoms_list = (
            new_data.get("exact_matches_symptom", [])
            + [s for s, _ in new_data.get("semantic_matches_symptom", [])]
        )
        vector = new_data["text_vector"]

        x = gnn_graph.x.clone().numpy()
        x[0] = [0, 1] + vector

        for disease in disease_list:
            if disease in self.alias2disease:
                x[self.node2id[self.alias2disease[disease]], 1] = 1
        
        edges = gnn_graph.edge_index.numpy()
        for symptom in symptoms_list:
            if symptom in self.node2id:
                x[self.node2id[symptom], 1] = 1
                symptom_id = self.node2id[symptom]
                edges = np.hstack([edges, np.array([[0], [symptom_id]])])
            

        gnn_graph.x = torch.tensor(x, dtype=torch.float)
        gnn_graph.edge_index = torch.tensor(edges, dtype=torch.long)
        
        y = np.zeros(len(self.disease_list), dtype=np.float32)
        for disease in list_y:
            if disease in self.disease2id:
                y[self.disease2id[disease]] = 1
        gnn_graph.y = torch.tensor(y, dtype=torch.float32)

        return gnn_graph

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]



if __name__ == "__main__":
    print('start')
    path_disease_symptom_csv = "data/disease_knowledge_graph/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    disease_symptom_csv =  pd.read_csv(path_disease_symptom_csv)
    
    main_graph_path = "data/disease_knowledge_graph/disease_csv_files/diseases_symptoms_merged.csv"
    disease_symptom_csv_path = "data/disease_knowledge_graph/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    patient_doctor_csv_path = "data/disease_knowledge_graph/patient-doctor.csv"
    disease_list_path = "data/disease_knowledge_graph/disease_csv_files/unique_aliases.csv"
    disease_aliases_path = "data/disease_knowledge_graph/disease_aliases.json"
    symptom_list_path = "data/disease_knowledge_graph/disease_csv_files/unique_symptoms.csv"
    qa_data_path = "data/disease_knowledge_graph/disease_csv_files/icliniq_medical_qa_cleaned.csv"
    disease_graphs = DiseaseKnowledgeGraph(main_graph_path, disease_symptom_csv_path, patient_doctor_csv_path, disease_list_path, symptom_list_path, disease_aliases_path, qa_data_path)

    with open("disease_graphs_qa.pkl", "wb") as f:
        pickle.dump(disease_graphs, f)

