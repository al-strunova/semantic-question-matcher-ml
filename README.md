# Question Pair Matcher

Find similar questions using semantic search with embeddings, FAISS indexing, and neural reranking.

## Features
- Semantic search using sentence transformers
- Fast similarity search with FAISS index
- Neural reranking for improved relevance
- Flask web interface and REST API
- Docker-ready deployment

## Prerequisites
- Python 3.11+
- Docker (optional, for containerization)

## Data Source
Uses the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs) containing over 400k question pairs.

## Getting Started

### Local Setup
1. **Clone**:
   ```sh
   git clone <repository-url>
   cd qqpmatcher
   ```

2. **Environment and Dependencies**:
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Flask**:
   ```sh
   python -m src.app
   ```

### Docker Deployment
1. **Build Image**:
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

## Rebuilding Index (Optional)
If you want to rebuild with fresh data:

1. **Download dataset**: Get `train.tsv` from [Kaggle QQP](https://www.kaggle.com/c/quora-question-pairs)
2. **Place in**: `data/qqp/train.tsv`  
3. **Rebuild**: `python scripts/build_index_offline.py`

## Architecture
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Search**: FAISS index for fast similarity search
- **Reranking**: Cross-encoder for relevance scoring
- **Storage**: Pickled DataFrames and binary index files

## Contributing
PRs are welcome. For major changes, open an issue first.

## License
MIT License.