#!/usr/bin/env python3
import os
import sys

def print_menu():
    print("=" * 50)
    print("      🧪 Science Article Control Panel")
    print("=" * 50)
    print(" [1] 📥 Extract All Data (Run Full Pipeline)")
    print(" [2] 🧬 Run Specific Data Extraction Step")
    print(" [3] 📊 Run Statistical Verification (math_statistics)")
    print(" [4] 📈 Generate Plots (math_statistics.plots)")
    print(" [5] 🧪 Run Tests (pytest)")
    print(" [6] ⚙️ Setup Environment (uv sync)")
    print(" [7] 🤖 Run ML Pipeline (Evaluate All Models)")
    print(" [8] 🧠 Train ResNet-18 (2D CNN)")
    print(" [9] 🔬 Run Ablation Study (Patch Sizes vs Features)")
    print(" [0] ❌ Exit")
    print("=" * 50)

def main():
    while True:
        os.system('clear')
        print_menu()
        try:
            choice = input("Select an option [0-6]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if choice == '1':
            os.system('bash scripts/run_all.sh')
        elif choice == '2':
            steps = sorted([f for f in os.listdir("scripts") if f.endswith(".sh") and not f.startswith("run_")])
            print("\nAvailable extractions:")
            for i, step in enumerate(steps, 1):
                print(f"  [{i:02d}] {step}")
            print("  [00] Back")
            subchoice = input("Select step: ").strip()
            if subchoice.isdigit() and 1 <= int(subchoice) <= len(steps):
                os.system(f'bash scripts/{steps[int(subchoice)-1]}')
        elif choice == '3':
            os.system('uv run python -m math_statistics.run_all')
        elif choice == '4':
            os.system('uv run python -m math_statistics.plots')
        elif choice == '5':
            os.system('bash scripts/run_tests.sh')
        elif choice == '6':
            os.system('bash scripts/00_setup.sh')
        elif choice == '7':
            os.system('PYTHONPATH=. ML/.venv/bin/python ML/evaluate_models.py')
        elif choice == '8':
            os.system('PYTHONPATH=. ML/.venv/bin/python ML/train_resnet.py')
        elif choice == '9':
            os.system('PYTHONPATH=. ML/.venv/bin/python ML/train_ablation.py')
        elif choice == '0':
            print("Bye!")
            break
        else:
            print("Invalid option. Please try again.")
            
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
