# Semantic Question Pair Matcher

Find similar questions using semantic search with embeddings, FAISS indexing, and neural reranking. 
This project demonstrates an end-to-end pipeline for building and deploying a semantic search microservice.

## Features
- Semantic search using state-of-the-art sentence transformers.
- Efficient candidate retrieval with a FAISS index.
- Relevance refinement using a Cross-Encoder neural reranker.
- Simple Flask web interface for demonstration.
- REST API for programmatic search.
- Docker-ready for containerized deployment.

## Prerequisites
- Python 3.11+
- Pip (Python package installer)
- Docker (optional, for containerization)

## Data Source
Uses the [Quora Question Pairs dataset](https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip)

## Getting Started

There are two ways to get the system running: using pre-computed artifacts (quickest) or rebuilding everything from scratch.

### Option 1: Using Pre-computed Artifacts (Recommended for Quick Evaluation)

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/al-strunova/semantic-question-matcher-ml.git
    cd semantic-question-matcher-ml
    ```

2.  **Create a virtual environment and install dependencies:**
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate 
    pip install -r requirements.txt
    ```

3.  **Download Pre-computed Artifacts:**
    The core search index and lookup data are required to run the service. Download the following files:
    * `questions_df.pkl`
    * `questions_index.faiss`
    * [link to download](https://drive.google.com/file/d/1TOf22ohkh7itZcDsDPxFkvsHMGVon04j/view?usp=drive_link)

    Create a `models` directory in the project root if it doesn't exist, and place the downloaded files into it:
    ```sh
    mkdir -p models
    # Move your downloaded files here, so you have:
    # models/questions_df.pkl
    # models/questions_index.faiss
    ```
    
4.  **Run the Flask Application:**
    ```sh
    python -m src.app
    ```
    The application will be available at `http://localhost:8080/`.

### Option 2: Rebuilding All Artifacts from Scratch

If you wish to rebuild the FAISS index and other artifacts from the raw data:

1.  **Clone and set up environment:** Follow steps 1 and 2 from "Option 1" above.
2.  **Download the Dataset:**
    * Download `train.tsv` from the [link](https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip).
    * Place the downloaded `train.tsv` file into the `data/qqp/` directory (create these directories if they don't exist):

3.  **Run the Build Script:**
    This script will process the data, generate embeddings, and build the FAISS index. This may take a significant amount of time depending on your hardware and if you're processing the full dataset.
    ```sh
    python scripts/build_index_offline.py
    ```
    This will create `questions_df.pkl` and `questions_index.faiss` in the `models/` directory.

4.  **Run the Flask Application:**
    ```sh
    python -m src.app
    ```

### Docker Deployment
1. **Prepare Artifacts:** Ensure `models/questions_df.pkl` and `models/questions_index.faiss` are present in your local `models/` directory
2. **Build Image**:
   ```sh
   docker build -t qqpmatcher .
   ```

2. **Run Container**:
   ```sh
   docker run -p 8080:5000 qqpmatcher
   ```

Open http://localhost:8080/ to access the search interface.

## Endpoints
- `/` : Main search page
- `/search` : Web interface for searching questions  
- `/api/search` : POST endpoint for programmatic search
- `/api/health` : Health check and model status

## API Usage
```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I learn Python?"}'
```

## Technical Architecture
-   **Embedding Model:** `sentence-transformers/all-mpnet-base-v2` used to generate dense vector representations of questions.
-   **Similarity Search Index:** FAISS (`IndexFlatL2`) for fast k-Nearest Neighbor search on embeddings.
-   **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` for refining relevance of candidates retrieved from FAISS.
-   **Backend:** Flask microservice.
-   **Data Storage (for lookup):** Pandas DataFrame serialized to a Pickle file (`questions_df.pkl`).


## Contributing
PRs are welcome. For major changes, open an issue first.

## License
MIT License.