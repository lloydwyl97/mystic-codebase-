import os
import sys
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

# Load environment variables (optional)
try:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    # If .env file doesn't exist or is corrupted, continue without it
    pass

# Import visualization modules
try:
    from chart_generator import plot_trades, plot_performance_over_time
    from mutation_graph import plot_strategy_graph
except ImportError as e:
    print(f"Error importing visualization modules: {e}")
    sys.exit(1)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return (
        jsonify(
            {
                "status": "healthy",
                "service": "visualization",
                "components": {
                    "chart_generator": "initialized",
                    "dashboard": "initialized",
                    "mutation_graph": "initialized",
                },
            }
        ),
        200,
    )


@app.route("/generate_trade_chart", methods=["POST"])
def generate_trade_chart():
    """Generate trade chart for a symbol"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "ETHUSDT")
        db_path = data.get("db_path", "simulation_trades.db")

        plot_trades(symbol, db_path)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Trade chart generated for {symbol}",
                    "file": "trade_chart.png",
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/generate_performance_chart", methods=["POST"])
def generate_performance_chart():
    """Generate performance chart"""
    try:
        data = request.get_json()
        db_path = data.get("db_path", "simulation_trades.db")

        plot_performance_over_time(db_path)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Performance chart generated",
                    "file": "performance_chart.png",
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/profit_chart_data", methods=["GET"])
def get_profit_chart_data():
    """Get profit chart data"""
    try:
        # This would typically load data from a database
        # For now, return a sample structure
        return (
            jsonify(
                {
                    "status": "success",
                    "data": {"times": [], "profits": [], "cumulative": []},
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/strategy_performance_data", methods=["GET"])
def get_strategy_performance_data():
    """Get strategy performance chart data"""
    try:
        # This would typically load data from a database
        # For now, return a sample structure
        return (
            jsonify(
                {
                    "status": "success",
                    "data": {
                        "strategies": [],
                        "win_rates": [],
                        "avg_profits": [],
                    },
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/mutation_graph", methods=["POST"])
def generate_mutation_graph():
    """Generate strategy mutation graph"""
    try:
        data = request.get_json()
        mutation_history = data.get("mutation_history", [])

        plot_strategy_graph(mutation_history)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Mutation graph generated",
                    "file": "mutation_graph.png",
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/chart/<filename>", methods=["GET"])
def get_chart_file(filename):
    """Serve generated chart files"""
    try:
        return send_file(filename, mimetype="image/png")
    except Exception:
        return (
            jsonify({"status": "error", "message": f"File {filename} not found"}),
            404,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8003))
    app.run(host="0.0.0.0", port=port, debug=False)
