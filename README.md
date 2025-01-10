# AlpacaFarm
AlpacaFarm: A Simulation Framework with Bayesian Optimization
AlpacaFarm is a simulation framework for methods that learn from human feedback, enabling researchers to rapidly iterate and evaluate models in a low-cost environment. This extended version integrates Bayesian Optimization to automate hyperparameter tuning, enhancing performance and simplifying model development.

Features
Simulated Feedback: Low-cost and efficient simulation of human feedback using API LLMs.
Reference Methods: Includes popular methods like PPO, DPO, and more for learning from pairwise feedback.
Bayesian Optimization: Automatically tunes hyperparameters using Gaussian Process Regression, replacing manual or grid-based searches.
Automatic Evaluation: Reliable performance measurement with simulated and human datasets.
Open Source: Fully open-source implementation to support research and development.
Getting Started
1. Prerequisites
Python 3.8 or later
Install required Python libraries:

pip install -r requirements.txt

2. Installation
Clone the repository:
git clone https://github.com/yourusername/alpaca_farm_bayesian.git
cd alpaca_farm_bayesian

Install dependencies:

pip install -r requirements.txt

3. Running the Project
Training Models with Bayesian Optimization
To train models using Bayesian optimization for hyperparameter tuning:

python train_with_bayesian_optimization.py

Evaluating Models
Run evaluations to compute win rates:

python evaluate_model.py
Hyperparameter Tuning
Modify the search space in train_with_bayesian_optimization.py:

space = [
    Real(1e-5, 1e-2, name="learning_rate"),
    Real(0.1, 10, name="kl_penalty"),
    Integer(16, 128, name="batch_size"),
]
Repository Structure

alpaca_farm_bayesian/
├── data/                       # Evaluation datasets
├── models/                     # Model configurations and checkpoints
├── scripts/
│   ├── train_with_bayesian_optimization.py  # Main script for training
│   ├── evaluate_model.py       # Script for evaluating models
│   └── utils.py                # Utility functions for training and evaluation
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
How Bayesian Optimization Works
In this project, Bayesian Optimization is used to tune key hyperparameters, such as:

Learning rate
KL penalty
Batch size
The framework uses scikit-optimize to:

Define a search space.
Minimize an objective function that evaluates model performance (e.g., win rate).
Select the best hyperparameters for final training.
Results
Using Bayesian optimization, we observed:

Improved model win rates compared to default hyperparameters.
Automated and efficient hyperparameter tuning without manual intervention.
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Stanford University for the AlpacaFarm framework.
Contributors for developing and maintaining the Bayesian optimization integration.
