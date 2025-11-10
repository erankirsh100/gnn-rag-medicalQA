FROM continuumio/miniconda3:latest

# avoid interactive prompts and expose conda env vars
ENV CONDA_ALWAYS_YES="true" \
    CONDA_ENV="conda_env" \
    CONDA_BIN="/opt/conda/bin/conda" \
    PATH="/opt/conda/envs/conda_env/bin:/opt/conda/bin:$PATH"

# define CONDA_PREFIX in a separate ENV so ${CONDA_ENV} is expanded
ENV CONDA_PREFIX="/opt/conda/envs/${CONDA_ENV}"


WORKDIR /app

# copy only environment file first to leverage layer caching
COPY setup/environment_linux.yml /app/environment.yml

# create the conda env at a known prefix so we can add it to PATH
RUN ${CONDA_BIN} env create -f /app/environment.yml -p ${CONDA_PREFIX} \
    && ${CONDA_BIN} clean -afy

# copy project
COPY pipeline.py /app/pipeline.py
COPY site/ /app/site/
COPY llm_files/ /app/llm_files/
COPY vector_db_files/searcher_class.py /app/vector_db_files/searcher_class.py
COPY data/ /app/data/
COPY .env /app/.env
COPY milvus_pmc.db /app/milvus_pmc.db
COPY gnn_files/best_model.pt /app/gnn_files/best_model.pt
COPY gnn_files/gnn.py /app/gnn_files/gnn.py
COPY gnn_files/utils.py /app/gnn_files/utils.py

# create entrypoint that activates the conda env for CMD/interactive shells
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'source /opt/conda/etc/profile.d/conda.sh || true' \
    'conda activate "${CONDA_ENV}" || true' \
    'exec "$@"' > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chmod 666 /app/milvus_pmc.db

# make sure scripts use bash
SHELL ["/bin/bash", "-lc"]

# expose port your Flask app uses
EXPOSE 5000

# ensure the conda env is active when container runs
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-u", "site/server.py"]