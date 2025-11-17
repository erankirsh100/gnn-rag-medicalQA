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

### Part 1 - Context Expansion
First, the user writes a complaint including his symptoms and additional known conditions. Then we expand appon what he wrote using 2 steps:

  #### A) GNN-based Desease Prediction <img align="right" src="utiles/gnn_pipeline.png" alt="Docker logo" width="110" style="vertical-align: right;"/>
  Based on past medical data collected from medical records of patients complaints and doctors diagnoses, we train a Graph Neural Network (GNN) model to predict possible deseases based on the symptoms provided by the user. Specifically, for each person in the training data we build a designated graph. The way we do it is we represent him as a node and connect him to his "symptom nodes" based on his description, and those symptom nodes to "deasese nodes" based on co-occurrence in the training data (this step is hard-coded and has additional minor details we skip here and get more into in the report). Then using deseases described in his doctor diagnosis as ground truth labels, we train a GNN model to predict possible deseases for new users based on their symptoms (the exact GNN architecture and training details are in the report).
  #### B) RAG-based Document Retrieval <img align="right" src="utiles/rag_pipeline.png" alt="Docker logo" width="110" style="vertical-align: right;"/>
  After we have a set of possible diseases from the GNN model, we use Retrieval-Augmented Generation (RAG) to retrieve relevant medical absracts from the <b>2016 Clinical Decision Support Track</b> dataset containing ~1.2M medical papers from PubMed Central tackling similar problem to ours - retrieving relevant medical papers based on users complaints. We took only the abstracts of the papers and additional inforamtion like title, authors, journal name etc. We built a vector database using PubMedBERT embeddings for vector representations, and IVF_FLAT index for fast retrieval (using k-means to devide the vector DB to 64 clusters and searching each time among the 10 closest ones for speed).
  Now, using the possible diseases from the GNN model as additional context, we query the vector database to retrieve relevant medical abstracts.

### Part 2 - Answer Generation
Using the user complaint, the deseases retrieved from the GNN and retrieved medical abstracts, we generate a final answer using a Large Language Model (LLM) - Google Gemini 1.5. The prompt is designed to provide the LLM with all the necessary context to generate an accurate and informative response.

<a name="about-the-code"></a>
## About The Code
In the end, it all comes down to the <b>pipeline.py</b> file. This file connects all the peieces together for a full run of the project.<br><br>
The main steps are:
1) acquire potential deseases using the GNN model with <code>graph_model.text_forward(user_input)</code>
2) retrieve relevant medical abstracts using RAG with <code>searcher.search(prompt, limit=5)</code>
3) generate the final answer using the LLM with <code>run_generation(query, graph_results, search_results_with_gnn)</code>

Of course, there are some additional optional parameters the functions get and these are used for testing purposes only (like providing ground truth diagnosis for the <code>run_generation</code> function to get desired evaluation metrics).



