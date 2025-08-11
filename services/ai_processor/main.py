import logging
import os
import sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Import AI modules
try:
    from ai_strategy_generator import AIStrategyGenerator
    from ai_genetic_algorithm import GeneticAlgorithmEngine
    from ai_model_versioning import ModelVersioningService
    from ai_auto_retrain import AutoRetrainService
except ImportError as e:
    logger.error(f"Error importing AI modules: {e}")
    sys.exit(1)

app = Flask(__name__)

# Initialize AI components
try:
    strategy_generator = AIStrategyGenerator()
    genetic_algorithm = GeneticAlgorithmEngine()
    model_versioning = ModelVersioningService()
    auto_retrain = AutoRetrainService()
except Exception as e:
    logger.error(f"Error initializing AI components: {e}")
    sys.exit(1)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return (
        jsonify(
            {
                "status": "healthy",
                "service": "ai_processor",
                "components": {
                    "strategy_generator": "initialized",
                    "genetic_algorithm": "initialized",
                    "model_versioning": "initialized",
                    "auto_retrain": "initialized",
                },
            }
        ),
        200,
    )


@app.route("/generate_strategy", methods=["POST"])
def generate_strategy():
    """Generate new AI trading strategy"""
    try:
        data = request.get_json()
        market_data = data.get("market_data", {})
        strategy_type = data.get("strategy_type", "breakout")

        strategy = strategy_generator.generate_strategy(market_data, strategy_type)
        return jsonify({"status": "success", "strategy": strategy}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/evolve_strategy", methods=["POST"])
def evolve_strategy():
    """Evolve existing strategy using genetic algorithm"""
    try:
        data = request.get_json()
        base_strategy = data.get("base_strategy", {})
        generations = data.get("generations", 10)

        evolved_strategy = genetic_algorithm.evolve_strategy(base_strategy, generations)
        return (
            jsonify({"status": "success", "evolved_strategy": evolved_strategy}),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/version_model", methods=["POST"])
def version_model():
    """Create new version of AI model"""
    try:
        data = request.get_json()
        model_data = data.get("model_data", {})
        version_notes = data.get("version_notes", "")

        version_info = model_versioning.create_version(model_data, version_notes)
        return (
            jsonify({"status": "success", "version_info": version_info}),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/auto_retrain", methods=["POST"])
def auto_retrain():
    """Trigger automatic model retraining"""
    try:
        data = request.get_json()
        training_data = data.get("training_data", {})
        model_id = data.get("model_id", "default")

        retrain_result = auto_retrain.retrain_model(training_data, model_id)
        return (
            jsonify({"status": "success", "retrain_result": retrain_result}),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    app.run(host="0.0.0.0", port=port, debug=False)
