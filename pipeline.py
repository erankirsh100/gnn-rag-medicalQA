import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils import encode_text_and_match_diseases, update_graph_data

graph_model = None
graph_data = None

def run_pipeline(query):
    encoded_dict = encode_text_and_match_diseases(query)
    related_graph_data = update_graph_data(graph_data, encoded_dict)
    graph_results = graph_model(related_graph_data.x, related_graph_data.edge_index, related_graph_data.edge_attr)
    
    return results

if __name__ == "__main__":
    query = " Hi, my name is XXXX I m a 19year old girl and I keep getting these weird movement feelings in the centre of my stomach (inside) it feel like there is something in there. Sometimes it give me a sharp pain doesn t really hurt just a quick weird pain. At first I thought I could be pregnant but then I tooktook a pregnancy test and it came up negative, and I am also using contraception ( the implant ) so I don t think I could be pregnant, and also the last time I had sex was 5 months ago I feel like I d no if I was pregnant 5 months gone. I just want to know what it is because it is a weird and slightly uncomfortable feeling because I don t know what it is or could be. Thankyou."
    results = run_pipeline(query)
    print("Generated Answer:", results)
