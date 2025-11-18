<img align="right" src="utiles/project_logo.png" alt="Project logo" width="120" />

# gnn-rag-medicalQA

- [Project Overview](#welcome-to-our-gnn-rag-medicalqa-project)👓
- [Quick start (raw project run)](#quick-start-raw-project-run)
  - [The Easy Way to Run the Project](#the-easy-way-to-run-the-project)🐳
  - [The Raw Project Run](#the-raw-project-run)🐍
- [How Does It Work?](#how-does-it-work)🤔
- [About The Code](#about-the-code)⚙️
- [Download / Build the Full Vector DB](vector_db_files/ReadMe.md#download-and-use)🗄️

<a name="welcome-to-our-gnn-rag-medicalqa-project"></a>
Welcome to our gnn-rag-medicalQA project!
The purpose of this project is to identify potential diseases from a user complaint that includes symptoms and any known conditions.

Platforms
- Supports Linux and Windows operating systems.
- Make sure to have <b>Docker installed</b> for the <b>easy run option</b>.
- Make sure to have <b>Conda</b> installed for the <b>raw project run option</b>. You may need to install Docker either way if you want to run on windows.

<div style="border:1px solid #e1e4e8; padding:12px 16px; border-radius:6px;">
  <strong>📝 Note</strong>
  <ul style="margin:8px 0 0 18px; padding:0;">
    <li>This repo uses a sample of ~15k medical papers from PubMed Central; our full dataset contains ~1.2M papers.</li>
    <li>Want the full experience? Replace <code>milvus_pmc.db</code> with the full DB available <a href="https://technionmail-my.sharepoint.com/:f:/g/personal/sasson_noam_campus_technion_ac_il/EnIMQ7Zc3E9OizbXFroOKwgBSqOaUzHcredlW4swTZNcaQ?e=ZssrXv" target="_blank" rel="noopener noreferrer">here</a>.</li>
  </ul>
</div>



<!-- Note
- The vector database can be used as a ready-made resource. You can skip building it. See vector_db_files/README.md for “Get a ready-to-use vector DB”. -->

<a name="quick-start-raw-project-run"></a>
## Quick start (raw project run)
### The Easy Way to Run the Project <img src="utiles/docker_icon.png" alt="Docker logo" width="35" style="vertical-align: middle;" />


1) Clone this gitHub repository
2) Change .env.example to .env and set your API keys (you need to set at least one, get one from the Gemini API: https://ai.google.dev/gemini-api/docs/api-key)
3) Run the following commands in your docker-enabled terminal:
```bash
docker build -t med_qa_app .
docker run -d --name med_qa -p 5000:5000 med_qa_app
```

4) Access the application at http://localhost:5000 in your web browser :)


<div style="border:1px solid #ffffffff; padding:12px 16px; border-radius:6px;">
<strong>📝 Note</strong>
<ul style="margin:8px 0 0 18px; padding:0;">
  <li>Build takes ~22 minutes on a standard machine.</li>
  <li>After running the <code>docker run</code> command, the server may take a minute or two to start responding so please be patient.</li>
</ul>
</div>

<a name="the-raw-project-run"></a>
### The Raw Project Run <img src="utiles/conda_icon.png" alt="Docker logo" width="35" style="vertical-align: middle;" />
1) Do steps 1,2 from the easy way.
2) Creating the Conda environment:
- Open an Anaconda/Miniconda shell.

  run from the project root:<br><br>
  Linux:
  ```bash
  conda env create -f setup/environment_linux.yml
  # activate the environment (replace <name> with the 'name' in the YAML if needed)
  conda activate <name>
  ```

  Windows:
  ```powershell
  conda env create -f setup\environment_windows.yml
  # activate the environment (replace <name> with the 'name' in the YAML if needed)
  conda activate <name>
  ```

3) Acquire the full vector database (not mandatory if using Linux)
- Read vector_db_files/README.md and follow the instructions for your OS to either:
  - Download and use the ready-to-use vector DB (recommended).
  - Build the Milvus vector DB from scratch (not recommended and supports Linux only).
- Ensure your .env has a proper PATH_TO_MILVUS_DB value as described there.

4) Runing the pipeline:
- After Milvus is up and reachable run from the project root:<br>

  Linux + Windows:
  ```bash
  python pipeline.py
  ```

Enjoy!

<a name="how-does-it-work"></a>
## How Does It Work?
<img style="vertical-align: middle;" src="utiles/pipeline.png" alt="Project logo" />

<br><br><br>
This project presents a robust, three-stage **knowledge-grounded clinical decision support pipeline**. Its primary objective is to transform a patient's free-text clinical query (symptoms and associated health concerns) into an **accurate, medically responsible, and highly interpretable structured response**. The system achieves this by integrating structured clinical knowledge via a Graph Neural Network (GNN) with comprehensive evidence retrieval using Retrieval-Augmented Generation (RAG).

### Part 1: Context Expansion and Evidence Elicitation

This initial phase systematically expands the user's raw input query by identifying plausible diagnoses and retrieving supporting evidence.

#### A) Knowledge Graph-Informed Diagnosis (GNN)

A Graph Neural Network (GNN) module is deployed to infer a ranked set of candidate diagnoses based on the reported symptoms.

* **Knowledge Graph Foundation:** A heterogeneous symptom-disease knowledge graph, derived from historical patient data, encodes clinically validated associations and co-occurrence patterns.
* **Patient-Specific Reasoning:** For a new query, the system dynamically inserts a **patient node**, connecting it to identified symptoms. This forms a specialized graph that allows the GNN to perform individualized reasoning.
* **Training and Prediction:** The GNN, implemented with a two-layer **GraphSAGE** model, is trained using the doctor's diagnoses as ground truth labels. During inference, the GNN propagates information across the graph structure to generate a probability distribution over potential diseases, yielding the top candidate diagnoses. This method ensures predictions are consistent with established clinical structure and empirical patient outcomes.

#### B) Evidence-Based Retrieval (RAG) 

A Retrieval-Augmented Generation (RAG) framework is used to surface relevant clinical evidence from biomedical literature.

* **Corpus:** The retrieval corpus consists of PubMed abstracts sourced from the **2016 Clinical Decision Support Track**.
* **Indexing and Search:** Documents are encoded into a semantic space using **PubMedBERT** for robust vector representations. These embeddings are stored in a vector database utilizing an **IVF\_FLAT** index for efficient Approximate Nearest-Neighbor (ANN) search. This index is optimized for speed by partitioning the database into **64 clusters** (via k-means) and searching only the 10 closest clusters for each query.
* **Graph-Informed Query Expansion:** To optimize retrieval accuracy, the input query is augmented by concatenating it with the candidate diagnoses from the GNN. This crucial step ensures the retrieval process is contextually rich and aligned with the structured diagnostic hypotheses. The resulting relevant medical abstracts form the evidence set.
* 
---

### Part 2: Controlled Diagnostic Response Generation

The final step synthesizes all compiled knowledge into a coherent clinical response.

* **Generative Model:** We utilize the **Google Gemini 2.5 Flash Lite** Large Language Model (LLM), selected for its efficient reasoning capability, operating under a controlled, low-temperature setting ($\tau=0.2$).
* **Prompt Engineering for Safety and Accuracy:** The LLM is conditioned on the original user query, the structured disease hypotheses, and the retrieved evidence. The prompt is meticulously engineered to enforce medically responsible generation, requiring the model to:
    * Formulate a cohesive **diagnostic hypothesis**.
    * Present transparent **clinical reasoning** that integrates both structured (GNN) and retrieved (RAG) evidence.
    * Recommend **responsible next steps** (e.g., monitoring, seeking professional consultation).
    * Conclude with a mandatory **safety disclaimer**.

---

<a name="about-the-code"></a>
## About The Code

The complete system orchestration is managed by the **pipeline.py** script, which seamlessly integrates the GNN, RAG, and LLM components.

The core execution flow involves three main function calls:

1) Acquisition of potential Diseases using the GNN model with <code>graph_model.text_forward(user_input)</code>
2) Retrieval of relevant bimedical evidence with <code>searcher.search(prompt, limit=5)</code>
3) Generation of the final answer using the LLM with <code>run_generation(query, graph_results, search_results_with_gnn)</code>



