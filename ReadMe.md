<img align="right" src="utiles/project_logo.png" alt="Project logo" width="120" />

# gnn-rag-medicalQA (Project Overview)
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