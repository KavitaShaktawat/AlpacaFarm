import numpy as np
import random

# Simulated Language Model
class LanguageModel:
    def __init__(self):
        self.weights = np.random.rand(10)  # Simple model with 10 parameters

    def generate(self, prompt):
        """Generate a response based on current weights."""
        return " ".join([prompt, f"response-{np.dot(self.weights, np.random.rand(10)):.2f}"])

    def update_weights(self, gradient, learning_rate=0.01):
        """Update weights based on gradient."""
        self.weights += learning_rate * gradient


# Simulated Human Feedback
class HumanFeedbackSimulator:
    def __init__(self, target_response="desired-response"):
        self.target_response = target_response

    def reward(self, response):
        """Simulate reward: Higher reward for closer matches to target_response."""
        return -abs(len(response) - len(self.target_response))  # Example reward based on string length


# Reinforcement Learning Trainer
class RLTrainer:
    def __init__(self, model, feedback_simulator):
        self.model = model
        self.feedback_simulator = feedback_simulator

    def compute_gradient(self, response, reward):
        """Compute a mock gradient based on reward."""
        return reward * np.random.rand(10)  # Simplified gradient computation

    def train(self, prompt, epochs=10):
        for epoch in range(epochs):
            response = self.model.generate(prompt)
            reward = self.feedback_simulator.reward(response)
            gradient = self.compute_gradient(response, reward)
            self.model.update_weights(gradient)
            print(f"Epoch {epoch + 1}, Reward: {reward}, Response: {response}")


# Evaluation
def evaluate_model(model, test_prompts):
    print("\nEvaluation Phase")
    for prompt in test_prompts:
        print(f"Prompt: {prompt} | Response: {model.generate(prompt)}")


# Main Framework Execution
if __name__ == "__main__":
    # Initialize components
    language_model = LanguageModel()
    feedback_simulator = HumanFeedbackSimulator()
    trainer = RLTrainer(language_model, feedback_simulator)

    # Training phase
    train_prompts = ["Hello", "How are you?", "Tell me a joke"]
    for prompt in train_prompts:
        trainer.train(prompt, epochs=5)

    # Evaluation phase
    test_prompts = ["What's the weather?", "Explain AI."]
    evaluate_model(language_model, test_prompts)
