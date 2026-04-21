import os
import subprocess
import sys

def run_command(command, description):
    print(f"{description}...")
    try:
        # Using sys.executable ensures the current environment is used
        subprocess.run([sys.executable] + command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failure during {description}: {e}")
        sys.exit(1)

def main():
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists("data/xview"):
        print("[ERROR] 'data/xview' directory not found. Please verify data paths.")
        return
    run_command(["src/train.py"], "Starting Model Training")

    if os.path.exists("outputs/model_epoch_2.pth"):
        run_command(["src/evaluate.py"], "Starting Quantitative Evaluation")
    else:
        print("[ERROR] Weights file not found in outputs/. Training may have failed.")
        return

    print("\n" + "="*50)
    print("Weights and metrics available in the 'outputs/' folder.")
    print("="*50)

if __name__ == "__main__":
    main()