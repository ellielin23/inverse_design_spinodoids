# auto_tune.py

from auto_train import main as train_main
from auto_evaluate import main as evaluate_main

def main():
    print("\n🚀 Starting auto-tuning...\n")
    train_main()
    print("\n✅ Training complete.\n")
    
    evaluate_main()
    print("\n✅ Evaluation complete.\n")

if __name__ == "__main__":
    main()
