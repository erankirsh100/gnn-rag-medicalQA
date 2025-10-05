import json
import re
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv




from utils import bert_encode  # your custom encoder
from sklearn.metrics.pairwise import cosine_similarity


torch_geometric.set_debug(True)


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Cache an embedding for "person" so we don't recompute each time
PERSON_VEC = bert_encode("person")


class DiseaseSymptomGraphGNN(nn.Module):
    """
    GraphSAGE model over a heterogeneous-like graph encoded as:
      - Node feature layout (first two slots are 'type flags' per your code):
          person:  [0, 1] + BERT(node_name)
          disease: [1, 0] + BERT(node_name)
          symptom: [2, 0] + BERT(node_name)
      - Edges:
          person -> all diseases
          disease -> symptom if present in CSV row
    Output: 1 logit per node (we will select only disease nodes via a mask).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dict_paths: Dict[str, str],
        hidden_channels: int,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5,
        with_bn: bool = True,
        seed: Optional[int] = None,
        disease_list_path: str = "data/disease_csv_files/unique_aliases.csv",
        symptom_list_path: str = "data/disease_csv_files/unique_symptoms.csv",
        disease_aliases_path: str = "data/disease_aliases.json",
    ):
        super().__init__()
        set_seed(seed)

        # Build a base graph from the provided disease-symptom dataframe
        self.main_graph, self.node2id, self.in_channels = self._create_graph(df)

        # Load disease/symptom metadata
        self.disease_list = pd.read_csv(disease_list_path)["0"].tolist()
        self.symptom_list = pd.read_csv(symptom_list_path)["symptoms"].tolist()

        # Load aliases (optional)
        self.disease2aliases = self._load_disease_aliases(disease_aliases_path)
        if self.disease2aliases:
            self.disease_list = sorted(df["diseases"].dropna().unique().tolist())
            self.alias2disease = {
                alias.lower(): disease
                for disease, aliases in self.disease2aliases.items()
                for alias in aliases
            }
        else:
            self.alias2disease = {}

        self.disease2id = {d: i for i, d in enumerate(self.disease_list)}

        self.main_graph_df = pd.read_csv(dict_paths["main_graph_path"])
        # self.disease_symptom_csv = pd.read_csv(dict_paths["disease_symptom_csv_path"])
        # self.patient_doctor_csv = pd.read_csv(dict_paths["patient_doctor_csv_path"])

        # Load disease list & mapping
        self.disease_list = sorted(self.main_graph_df["diseases"].dropna().unique().tolist())
        print(f'Number of unique diseases: {len(self.disease_list)}')
        self.disease2id = {disease: i for i, disease in enumerate(self.disease_list)}

        # Load symptoms
        self.symptom_list = pd.read_csv(symptom_list_path)["symptoms"].tolist()
        self.alias_disease_list = pd.read_csv(disease_list_path)["0"].tolist()

        # Load disease aliases
        self.disease2aliases = self._load_disease_aliases(dict_paths["disease_aliases_path"])
        if self.disease2aliases:
            self.alias2disease = {
                alias.lower(): disease
                for disease, aliases in self.disease2aliases.items()
                for alias in aliases
            }
        else:
            self.alias2disease = {}

        # Boolean mask for disease nodes on this *base graph* (useful for inference on self.main_graph)
        num = 0
        self.model_graph_disease_mask = torch.zeros(self.main_graph.num_nodes, dtype=torch.bool)
        for d in self.disease_list:
            if d in self.node2id:
                self.model_graph_disease_mask[self.node2id[d]] = True
                num += 1
        
        print(f'Number of disease nodes in model graph: {num}')

        # ---- Define GNN ----
        assert num_layers >= 2, "Use at least 2 GraphSAGE layers."
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(self.in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        if with_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final layer to a single logit per node (for BCEWithLogits)
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.with_bn = with_bn
        self.act = F.relu

    @staticmethod
    def _load_disease_aliases(path: Optional[str]) -> Dict[str, List[str]]:
        if path is None:
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _create_graph(self, df: pd.DataFrame) -> Tuple[Data, Dict[str, int], int]:
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

    # ------- Optional text->graph helper for inference -------
    def encode_text_and_match_diseases(self, text: str, top_k: int = 5) -> Dict:
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

    def forward_pass(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # GraphSAGE forward without edge weights
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)            # <- no edge_attr here
            if self.with_bn:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x  # raw logits (no activation)
    
    def text_forward(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Run text through the model and return top_k diseases with highest probability.
        """
        self.eval()
        with torch.no_grad():
            encoding_dict = self.encode_text_and_match_diseases(text, top_k=top_k)
            updated_graph = self.update_graph_data(encoding_dict, list_y=[])
            logits = self.forward_pass(updated_graph.x, updated_graph.edge_index)
            
            # Extract only disease logits
            disease_logits = logits[updated_graph.disease_mask].squeeze()  # [num_diseases]
            
            # Convert to probabilities (softmax so they sum to 1)
            probs = torch.softmax(disease_logits, dim=0).cpu().numpy()
            
            # Get top_k indices
            top_indices = probs.argsort()[-top_k:][::-1]
            
            # Map back to disease names
            top_diseases = [self.disease_list[i] for i in top_indices]

        return top_diseases

    # Simple convenience forward for the model's *internal* base graph
    def forward(self) -> torch.Tensor:
        return self.forward_pass(self.main_graph.x, self.main_graph.edge_index)
