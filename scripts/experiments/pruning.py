import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import copy
from torch.nn.utils import prune
import matplotlib.pyplot as plt
import os
os.makedirs("results", exist_ok=True)

from scripts.trigger import TriggerGenerator, generate_trigger_graph
from scripts.train import bi_level_optimization
from scripts.metrics import load_dataset
from models_def.gcn import GCN
from models_def.gat import GAT
from models_def.graphsage import GraphSAGE



def run_pruning_experiment():
    datasets = ['Cora', 'Pubmed']
    models = {
        'GCN': lambda d: GCN(d.num_features, 128, d.num_classes),
        'GAT': lambda d: GAT(d.num_features, 64, d.num_classes),
        'GraphSAGE': lambda d: GraphSAGE(d.num_features, 128, d.num_classes)
    }

    pruning_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {model: {dataset: [] for dataset in datasets} for model in models.keys()}

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        data = dataset[0]

        indices = torch.randperm(data.num_nodes)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[indices[:int(0.6*data.num_nodes)]] = True
        data.val_mask[indices[int(0.6*data.num_nodes):int(0.8*data.num_nodes)]] = True
        data.test_mask[indices[int(0.8*data.num_nodes):]] = True

        for model_name, model_fn in models.items():
            generator = TriggerGenerator(dataset.num_features, 64, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
            wm_model = model_fn(dataset)
            bi_level_optimization(wm_model, generator, data)
            original_state = copy.deepcopy(wm_model.state_dict())

            for rate in pruning_rates:
                wm_model = model_fn(dataset)
                wm_model.load_state_dict(original_state)

                parameters_to_prune = []
                for name, module in wm_model.named_modules():
                    if isinstance(module, (GCNConv, GATConv, SAGEConv, nn.Linear)):
                        if hasattr(module, 'weight'):
                            parameters_to_prune.append((module, 'weight'))

                if parameters_to_prune:
                    for module, param_name in parameters_to_prune:
                        prune.l1_unstructured(module, name=param_name, amount=rate)
                        prune.remove(module, param_name) 

                # Regenerate trigger graph AFTER pruning
                trigger_data = generate_trigger_graph(data, generator, wm_model)

                wm_model.eval()
                with torch.no_grad():
                    pred = wm_model(trigger_data.x, trigger_data.edge_index).argmax(dim=1)
                    if hasattr(trigger_data, 'trigger_nodes'):
                        wm_mask = torch.zeros(trigger_data.num_nodes, dtype=torch.bool)
                        wm_mask[trigger_data.trigger_nodes] = True
                        wm_acc = (pred[wm_mask] == trigger_data.y[wm_mask]).float().mean().item() * 100
                        results[model_name][dataset_name].append(wm_acc)

    print("\nWatermark accuracy at different pruning rates (%):")
    print("| Model     | Dataset |  0%  | 10%  | 30%  | 50%  | 70%  | 90%  |")
    print("|-----------|---------|------|------|------|------|------|")
    for model_name in models.keys():
        for dataset in datasets:
            accs = results[model_name][dataset]
            print(f"| {model_name:<9} | {dataset:<7} | {accs[0]:4.1f} | {accs[1]:4.1f} | {accs[3]:4.1f} | {accs[5]:4.1f} | {accs[7]:4.1f} | {accs[9]:4.1f} |")

    # Plot
    plt.figure(figsize=(15, 4))
    for i, model_name in enumerate(models.keys()):
        plt.subplot(1, 3, i+1)
        for dataset in datasets:
            plt.plot(
                [p * 100 for p in pruning_rates],
                results[model_name][dataset],
                marker='o', linestyle='-', label=dataset
            )
        plt.title(f'({chr(97+i)}) {model_name}')
        plt.xlabel('Pruning Rate (%)')
        plt.ylabel('Watermark Accuracy (%)')
        plt.ylim(50, 100)
        plt.tick_params(top=False, right=False)
        if i == 0:
            plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/pruning_robustness.png')
    # plt.show()







if __name__ == "__main__":
    run_pruning_experiment()