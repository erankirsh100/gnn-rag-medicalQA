<img align="right" src="utiles/project_logo.png" alt="Project logo" width="120" />

# gnn-rag-medicalQA (Project Overview)

The purpose of this project is to identify potential diseases from a user complaint that includes symptoms and any known conditions.

Platforms
- Supports Linux and Windows operating systems.
- Conda environment is mandatory, please install Miniconda or Anaconda if not already installed.

Note
- The vector database can be used as a ready-made resource. You can skip building it. See vector_db_files/README.md for “Get a ready-to-use vector DB”.

## Quick start (raw project run)

1) Get the code
- Clone or download this repository to your local machine (Linux or Windows).

2) Create the Conda environment
- Open an Anaconda/Miniconda shell.

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

3) Acquire the vector database (required for running)
- Read vector_db_files/README.md and follow the instructions for your OS to either:
  - Build the Milvus vector DB, or
  - Download and use the ready-to-use vector DB (recommended for speed).
- Ensure your .env has a proper PATH_TO_MILVUS_DB value as described there.

4) Run the pipeline
- After Milvus is up and reachable (per the previous step), run the main pipeline from the project root.

Linux:
```bash
python3 pipeline.py
```

Windows:
```powershell
python .\pipeline.py
```

Enjoy!