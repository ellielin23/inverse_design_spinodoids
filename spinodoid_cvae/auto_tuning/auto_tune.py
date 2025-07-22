# auto_tune.py

# this file is used to run the auto-tuning process
# when ready to run, move all files to the root directory

from auto_train import main as train_main
from auto_evaluate import main as evaluate_main

def main():
    print("\n✅ Starting auto-tuning...\n")
    train_main()
    print("\n✅ Training complete.\n")
    
    evaluate_main()
    print("\n✅ Evaluation complete.\n")

if __name__ == "__main__":
    main()
