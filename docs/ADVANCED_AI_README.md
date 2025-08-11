# 🧠 Advanced AI Trading Platform

## Overview

The Mystic Trading Platform now includes comprehensive advanced AI capabilities for reinforcement learning, genetic algorithms, neural networks, and automated strategy evolution. This system is designed to run on Windows Home 11 with Docker support.

## 🚀 Quick Start

### Prerequisites

- Windows Home 11
- Python 3.11+
- Docker Desktop for Windows
- PowerShell

### Installation

1. **Run the Advanced AI Setup Script:**

   ```powershell
   .\setup-advanced-ai.ps1
   ```

2. **Start Advanced AI Services:**

   ```powershell
   .\start-advanced-ai-docker.ps1
   ```

3. **Access Dashboards:**
   - MLflow Tracking: <http://localhost:5000>
   - Streamlit Dashboard: <http://localhost:8501>
   - Dash Dashboard: <http://localhost:8050>
   - Optuna Dashboard: <http://localhost:8080>
   - TensorBoard: <http://localhost:6006>
   - Flower (Celery): <http://localhost:5555>
   - Ray Dashboard: <http://localhost:8265>

## 🧠 Advanced AI Features

### 1. Reinforcement Learning (RL)

- **Ray RLlib**: Distributed RL training
- **Stable-Baselines3**: State-of-the-art RL algorithms
- **Gymnasium**: RL environments
- **Algorithms**: PPO, A2C, DQN, SAC, TD3

### 2. Genetic Algorithms (GA)

- **DEAP**: Evolutionary algorithms
- **NEAT**: Neuroevolution
- **PyGAD**: Genetic algorithm library
- **PyMOO**: Multi-objective optimization

### 3. Neural Networks & Deep Learning

- **PyTorch**: Deep learning framework
- **TensorFlow**: Alternative framework
- **Transformers**: State-of-the-art NLP models
- **DeepSpeed**: High-performance training

### 4. Strategy Evolution

- **Automated Strategy Generation**: AI creates new strategies
- **Genetic Algorithm Optimization**: Evolves strategy parameters
- **Multi-Objective Optimization**: Balances risk vs return
- **Strategy Mutation**: Intelligent strategy modification

### 5. Model Management

- **MLflow**: Experiment tracking and model versioning
- **ONNX**: Model export and sharing
- **Joblib**: Model serialization
- **TensorFlow Serving**: Model deployment

### 6. Hyperparameter Optimization

- **Optuna**: Advanced hyperparameter optimization
- **Bayesian Optimization**: Efficient parameter search
- **Multi-Objective Optimization**: Multiple optimization targets

## 📊 Web Dashboards

### Streamlit Dashboard

- Real-time AI model performance
- Strategy evolution visualization
- Interactive parameter tuning
- Live trading signals

### Dash Dashboard

- Advanced charting and analytics
- Model comparison tools
- Performance metrics
- Risk analysis

### MLflow Tracking

- Experiment tracking
- Model versioning
- Performance comparison
- Artifact management

### Optuna Dashboard

- Hyperparameter optimization progress
- Trial visualization
- Parameter importance analysis
- Optimization history

## 🔧 Configuration

### Advanced AI Configuration (`advanced_ai_config.ini`)

```ini
[AI_SETTINGS]
# Reinforcement Learning
enable_rl_training = true
rl_algorithm = stable_baselines3
ray_cluster_mode = false

# Strategy Evolution
enable_genetic_algorithms = true
mutation_rate = 0.1
population_size = 100
generations = 50

# Model Export
enable_model_export = true
export_formats = onnx,tensorflow,joblib
model_versioning = true

# Web UI Dashboards
enable_dash_dashboards = true
enable_streamlit_apps = true
dashboard_port = 8050
streamlit_port = 8501

# Performance Optimization
enable_deepspeed = true
mixed_precision = true
gradient_accumulation = 4

# Experiment Tracking
enable_mlflow = true
enable_wandb = false
mlflow_tracking_uri = sqlite:///mlflow.db

# Hyperparameter Optimization
enable_optuna = true
n_trials = 100
optimization_direction = maximize
```

## 📁 Directory Structure

```text
Mystic-Codebase/
├── mlflow/                 # MLflow tracking data
├── ray/                   # Ray cluster data
├── optuna/                # Optuna optimization data
├── models/                # Trained models
├── logs/                  # Training logs
├── wandb/                 # Weights & Biases data
├── strategies/            # AI-generated strategies
│   ├── rl_basic_strategy.json
│   └── ga_optimized_strategy.json
├── frontend/              # Dashboard applications
├── backend/               # Backend services
└── docker-compose-advanced-ai.yml
```

## 🚀 Usage Examples

### 1. Training a Reinforcement Learning Model

```python
from stable_baselines3 import PPO
from gymnasium import make

# Create environment
env = make("trading_env_v1")

# Train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# Save model
model.save("models/ppo_trading_model")
```

### 2. Genetic Algorithm Strategy Optimization

```python
import deap
from deap import base, creator, tools, algorithms

# Define genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Setup toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evolution
population = toolbox.population(n=100)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3,
                   ngen=50, verbose=True)
```

### 3. Hyperparameter Optimization with Optuna

```python
import optuna

def objective(trial):
    # Define hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Train model and return performance
    model = train_model(learning_rate, batch_size)
    return evaluate_model(model)

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

## 🔄 Strategy Evolution Process

1. **Initial Population**: Generate diverse trading strategies
2. **Evaluation**: Test strategies on historical data
3. **Selection**: Choose best-performing strategies
4. **Crossover**: Combine successful strategies
5. **Mutation**: Introduce random variations
6. **Replacement**: Update population with new strategies
7. **Repeat**: Continue evolution for specified generations

## 📈 Performance Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Total Return**: Absolute performance
- **Maximum Drawdown**: Risk measurement
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return vs maximum drawdown
- **Sortino Ratio**: Downside risk-adjusted returns

## 🛠️ Management Commands

### Docker Services

```powershell
# Start services
docker-compose -f docker-compose-advanced-ai.yml up -d

# Stop services
docker-compose -f docker-compose-advanced-ai.yml down

# View logs
docker-compose -f docker-compose-advanced-ai.yml logs -f

# Restart services
docker-compose -f docker-compose-advanced-ai.yml restart
```

### Individual Services

```powershell
# MLflow
docker run -p 5000:5000 -v ./mlflow:/mlflow python:3.11-slim

# Ray
docker run -p 8265:8265 -v ./ray:/ray python:3.11-slim

# Streamlit
docker run -p 8501:8501 -v ./frontend:/app python:3.11-slim
```

## 🔍 Troubleshooting

### Common Issues

1. **Docker not starting**
   - Ensure Docker Desktop is installed and running
   - Check Windows WSL2 is enabled
   - Restart Docker Desktop

2. **Port conflicts**
   - Check if ports are already in use
   - Modify ports in docker-compose-advanced-ai.yml
   - Use `netstat -ano` to find conflicting processes

3. **Memory issues**
   - Increase Docker memory limit in Docker Desktop settings
   - Reduce batch sizes in training configurations
   - Use CPU-only versions of PyTorch/TensorFlow

4. **CUDA issues**
   - Install NVIDIA drivers
   - Use CPU-only versions if GPU not available
   - Check CUDA compatibility

### Performance Optimization

1. **GPU Acceleration**
   - Install CUDA toolkit
   - Use GPU-enabled Docker images
   - Enable mixed precision training

2. **Distributed Training**
   - Use Ray cluster for distributed RL
   - Enable multi-GPU training
   - Use DeepSpeed for large models

3. **Memory Management**
   - Use gradient accumulation
   - Implement model checkpointing
   - Optimize batch sizes

## 📚 Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Optuna Documentation](https://optuna.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your AI enhancements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎉 Your Mystic Trading Platform is now equipped with cutting-edge AI capabilities!**

For support and questions, please refer to the main README.md or create an issue in the repository.
