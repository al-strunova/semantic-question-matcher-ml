from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from .config import Config
from .embedding import load_embedding_model
from .faiss_index_search import load_faiss_index
from .reranker import load_reranker
from .question_matcher import search_similar_questions


def create_app(config_object=Config):
    # Get absolute path to templates directory (one level up from src)
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))

    app = Flask(__name__, template_folder=template_dir)
    app.config.from_object(config_object)

    # Store models in app instance (avoiding globals)
    app.embedding_model = None
    app.faiss_index = None
    app.questions_df = None
    app.reranker = None

    # Load models immediately
    load_models(app)

    @app.route('/')
    def index():
        """Render the search page"""
        return render_template('index.html')

    @app.route('/api/search', methods=['POST'])
    def api_search():
        """API endpoint for searching similar questions"""
        try:
            # Get query from request
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({'error': 'No query provided'}), 400

            query = data['query']

            # Check if models are loaded
            if not all([app.embedding_model, app.faiss_index, app.questions_df, app.reranker]):
                return jsonify({'error': 'Models not loaded. Please try again later.'}), 503

            # Find similar questions
            results = search_similar_questions(
                query,
                app.embedding_model,
                app.faiss_index,
                app.questions_df,
                app.reranker
            )

            return jsonify({'results': results})

        except Exception as e:
            print(f"API error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/search')
    def search_page():
        """Web interface for searching"""
        query = request.args.get('query', '')
        print(f"Received search query: {query}")  # Debug print

        results = []

        if query:
            try:
                # Check if models are loaded
                if (app.embedding_model is None or app.faiss_index is None or app.questions_df is None
                        or app.reranker is None):
                    print("Models not fully loaded")  # Debug print
                    return render_template('error.html',
                                           error="Search system is initializing. Please try again later.")

                # Find similar questions
                print("Searching for similar questions...")  # Debug print
                results = search_similar_questions(
                    query,
                    app.embedding_model,
                    app.faiss_index,
                    app.questions_df,
                    app.reranker
                )
                print(f"Found {len(results)} results")
                # Add more debugging
                print(f"First result: {results[0] if results else 'No results'}")

            except Exception as e:
                print(f"Error during search: {str(e)}")  # Debug print
                return render_template('error.html', error=str(e))

        # If no query or after getting results
        return render_template('results.html', query=query, results=results)

    @app.route('/api/health')
    def health_check():
        """Health check endpoint for monitoring"""
        models_loaded = all([
            app.embedding_model,
            app.faiss_index,
            app.questions_df,
            app.reranker
        ])

        return jsonify({
            "status": "ready" if models_loaded else "initializing",
            "version": Config.VERSION
        })

    return app


def load_models(app):
    """Load all models and data"""
    print("Loading models and data...")

    try:
        # Check if required files exist
        if not os.path.exists(Config.QUESTIONS_DF_PATH):
            print(f"Error: Questions DataFrame not found at {Config.QUESTIONS_DF_PATH}")
            print("Please run build_index_offline.py first")
            return

        if not os.path.exists(Config.FAISS_INDEX_PATH):
            print(f"Error: FAISS index not found at {Config.FAISS_INDEX_PATH}")
            print("Please run build_index_offline.py first")
            return

        # Load models
        app.embedding_model = load_embedding_model()
        print("✓ Embedding model loaded")

        app.reranker = load_reranker()
        print("✓ Reranker loaded")

        app.questions_df = pd.read_pickle(Config.QUESTIONS_DF_PATH)
        print(f"✓ Questions DataFrame loaded with {len(app.questions_df)} entries")

        app.faiss_index = load_faiss_index()
        print("✓ FAISS index loaded")

        print("All models and data loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")


# This would go in main.py or can be in the same file if preferred
if __name__ == '__main__':
    # Create and run the app
    app = create_app()
    app.run(debug=True)