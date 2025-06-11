from scripts.experiments.comparison import run_comparison
from scripts.experiments.pruning import run_pruning_experiment
from scripts.experiments.finetune import run_finetune_experiment
from scripts.experiments.evasion import run_evasion_attack
from scripts.experiments.fraud import run_fraudulent_declaration

if __name__ == "__main__":
    run_comparison()
    run_pruning_experiment()
    run_finetune_experiment()
    run_evasion_attack()
    run_fraudulent_declaration()
