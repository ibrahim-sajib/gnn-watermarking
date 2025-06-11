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
gnn_watermarking/
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


## References
   Zhang, Linji, Mingfu Xue, Leo Yu Zhang, Yushu Zhang, and Weiqiang Liu. "An imperceptible and owner-unique watermarking method for graph neural networks." In Proceedings of the ACM Turing Award Celebration Conference-China 2024, pp. 108-113. 2024.