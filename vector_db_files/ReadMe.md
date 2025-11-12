# Vector Database: Usage and Creation Guide
Supported platforms: Windows and Linux.

We present to you 2 options:<br>
* The easy way😇: download and use our ready-to-use Milvus vector DB! (recommended)
* The hard way😈: build the Milvus vector DB from scratch! (not recommended unless you have sufficient memory, compute power, and time)


# Download and Use

### Linux
- Download the premade `milvus_pmc.db`: [Download Link](https://technionmail-my.sharepoint.com/:u:/g/personal/sasson_noam_campus_technion_ac_il/EbDphKAmJWBDm8CP2BIclYYB7jX6p2JsKeyN_7Jt6hNvQQ?e=iwIbIm)
- Place it in the project root (or your chosen folder).
- Set `.env`:
```
PATH_TO_MILVUS_DB=./milvus_pmc.db
```

### Windows
- Download the prepared data from ADD_LINK_HERE to destination setup\vector_db_setup\windows.
- run the following in your docker-enabled terminal:
  ```powershell
  mkdir milvus && cd milvus
  Invoke-WebRequest -Uri https://raw.githubusercontent.com/milvus-io/milvus/v2.4.6/deployments/docker/compose/standalone/docker-compose.yml -OutFile docker-compose.yml
  docker compose up -d
  docker ps  # verify milvus-standalone is running
  ```
- Ensure Docker/Milvus is running and `.env` has `PATH_TO_MILVUS_DB=http://127.0.0.1:19530`.
- Open and run the following notebook: setup/vector_db_setup/windows/<b>upload_data_to_docker.ipynb</b> with the provided conda kernel (check ReadMe in root if you haven't set it up yet).

## Notes and tips

- .env formatting: use simple KEY=VALUE, for example:
  ```
  PATH_TO_MILVUS_DB=http://127.0.0.1:19530
  ```
  No quotes are required.
- On Windows, run shell scripts with Git Bash or WSL.
<br><br>
# Build the Milvus Vector DB
## Overview
This guide covers:
- Preparing the TREC Clinical (2016) PMC data
- Converting NXML to compact JSON
- Building a Milvus vector database (Windows via Docker, Linux via Milvus Lite)

Result: a Milvus vector DB usable by the pipeline.

## Prerequisites

- Python 3.9+ and pip
- Jupyter (e.g., VS Code + Python extension or `pip install jupyter`)
- tqdm, transformers, torch, python-dotenv, pymilvus (installed by the notebook as needed)
- Disk space: tens of GB if processing all PMC archives

## 1) Download and extract the dataset

1. Download the four PMC archives (pmc-00…pmc-03) from:
   https://trec.nist.gov/data/clinical2016.html

2. Extract the NXML files using the script:
   - Place the downloaded .tar.gz files in a folder you choose.
   - Run the extractor from that folder:

   Linux/macOS + Windows:
   ```bash
   bash path/to/your/repo/vector_db_files/extract_data_from_zip.sh
   ```


   - Alternatively, extract manually ensuring you end up with:
     - raw_data/pmc-00, raw_data/pmc-01, raw_data/pmc-02, raw_data/pmc-03
     - res_data/ (empty to start)

## 2) Build the vector DB

The notebook `vector_db_files/vector_db_creation.ipynb` converts PMC NXML to compact JSON and indexes embeddings into Milvus.

Note about paths:
- The notebook has example Linux paths like `/home/student/project/...`.
- Replace these with your local paths. Example (Windows): `C:\Users\Lenovo\Documents\...\raw_data` and `...\res_data`.

### Windows (Docker Milvus)
unfo

1) Install Docker Desktop for Windows
- Download and install: https://www.docker.com/products/docker-desktop/
- Enable WSL 2 during installation if prompted.
- Start Docker Desktop and ensure it’s running.

2) Start Milvus Standalone (recommended via Docker Compose)
PowerShell:
```powershell
mkdir milvus && cd milvus
Invoke-WebRequest -Uri https://raw.githubusercontent.com/milvus-io/milvus/v2.4.6/deployments/docker/compose/standalone/docker-compose.yml -OutFile docker-compose.yml
docker compose up -d
docker ps  # verify milvus-standalone is running
```

3) Configure .env
```
PATH_TO_MILVUS_DB=http://127.0.0.1:19530
```

4) Run the notebook
- Open `vector_db_files/vector_db_creation.ipynb`
- Update the data paths in the cells to your local folders
- Run all cells. Ensure Docker/Milvus is running.

### Linux (Milvus Lite, no Docker)

1) Configure .env
```
PATH_TO_MILVUS_DB=./milvus_pmc.db
```

2) Run the notebook
- Open `vector_db_files/vector_db_creation.ipynb`
- Update the data paths in the cells to your local folders
- Run all cells. A `milvus_pmc.db` file will be created in your working directory (Milvus Lite).

## 3) Verify

- Windows (Docker): `docker ps` should show milvus running; the notebook should complete inserts without errors.
- Linux (Lite): `milvus_pmc.db` should appear and grow in size during insertion.

Optional Python check:
```python
from pymilvus import MilvusClient
import os
client = MilvusClient(uri=os.getenv("PATH_TO_MILVUS_DB"))
print(client.list_collections())
```
