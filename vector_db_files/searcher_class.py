from pymilvus import MilvusClient
from transformers import AutoTokenizer, AutoModel
import torch

class MilvusSearcher:
    def __init__(self, uri: str, collection_name: str):
        self.milvus_client = MilvusClient(uri=uri)
        self.collection_name = collection_name

        # Check collection existence
        if not self.milvus_client.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' does not exist.")
            return
        else:
            print(f"Collection '{self.collection_name}' exists.")

        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO")
        self.model = AutoModel.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO").to(self.device)
        self.model.to(self.device)

    def encode_text(self, title, abstract):
        """Encode text using PubMedBERT with GPU support."""
        margin = 12
        max_length = 512 - margin # Maximum length for PubMedBERT
        text = f"{title} {abstract}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
    
        # Move embeddings back to CPU for numpy conversion
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings

    def search(self, query: str, limit: int = 10):
        query_vector = self.encode_text(query, "")
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            output_fields=["doc"],
        )
        return search_res