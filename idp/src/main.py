# src/main.py

from evaluate import evaluate_model
from train import train_model
from interpret import interpret_model

def main():
    """
    Main function to run the full ML pipeline:
    1. Train the model.
    2. Interpret the model's predictions.
    """
    print("===== Running Full ML Pipeline =====")
    train_model()
    evaluate_model()
    interpret_model()
    print("===== Pipeline Finished Successfully =====")

if __name__ == '__main__':
    main()