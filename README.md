# GNN Watermarking Project

This project implements a robust, imperceptible watermarking method for Graph Neural Networks (GNNs), as described in the paper _"An Imperceptible and Owner-unique Watermarking Method for Graph Neural Networks"_.

## Features

- Train and evaluate watermarked GNNs
- Compare original and watermarked model metrics
- Test robustness under:
  - Fine-tuning
  - Model pruning
  - Evasion attacks
  - Fraudulent declarations
- ROC curve generation for trigger node detectability



## Repository Structure

```plaintext  
gnn-watermarking/
├── main.py                     # Entry point (to run experiments)
├── models/                     # Trained model .pth files
|-- notebooks/                  # ipynb File
├── scripts/
│   ├── train.py                # train_model()
│   ├── trigger.py             # TriggerGenerator, generate_trigger_graph()
│   ├── metrics.py             # calculate_metrics()
│   └── experiments/
│       ├── comparison.py      # run_comparison()
│       ├── pruning.py         # run_pruning_experiment()
│       ├── finetune.py        # run_finetune_experiment()
│       ├── evasion.py         # run_evasion_attack()
│       ├── fraud.py           # run_fraudulent_declaration()
├── models_def/
│   ├── gcn.py                 # GCN model
│   ├── gat.py                 # GAT model
│   ├── sage.py                # GraphSAGE model
├── requirements.txt
└── README.md
```



# How to replicate

1. Navigate to your project folder
```
cd your_project_directory
```
2. Create the virtual environment
```
python -m venv .venv
```
3. Activate the virtual environment
```
source .venv/bin/activate
```
4. Install dependencis
```
pip install -r requirements.txt
```

5. Run the code
```
python main.py
```

# 📊 Run Experiments Individually
Each script replicates a specific result from the paper. Outputs are saved in the results/ directory.

6. Table 2 & Table 3 — Model Fidelity and Watermark Accuracy
```
python -m scripts.experiments.comparison
```
Output:
results/model_performance_table.csv
results/watermark_accuracy_table.csv

7. Figure 2 — Fine-Tuning Robustness
```
python -m scripts.experiments.finetune
```
Output:
results/fine_tuning_robustness.png

8. Figure 3 — Pruning Robustness
```
python -m scripts.experiments.pruning
```
Output:
results/pruning_robustness.png

9. Figure 4 — Evasion Attack (ROC Curve)
```
python -m scripts.experiments.evasion
```
Output:
results/evasion_roc_curve.png

10. Figure 5 — Fraudulent Ownership Attack
```
python -m scripts.experiments.fraud
```
Output:
results/fraudulent_declaration.png




## References
   Zhang, Linji, Mingfu Xue, Leo Yu Zhang, Yushu Zhang, and Weiqiang Liu. "An imperceptible and owner-unique watermarking method for graph neural networks." In Proceedings of the ACM Turing Award Celebration Conference-China 2024, pp. 108-113. 2024.