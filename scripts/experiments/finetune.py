import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scripts.trigger import TriggerGenerator, generate_trigger_graph
from scripts.train import bi_level_optimization
from scripts.metrics import load_dataset
from models_def.gcn import GCN
from models_def.gat import GAT
from models_def.graphsage import GraphSAGE


def run_finetune_experiment():
    datasets = ['Cora', 'Pubmed']
    models = {
        'GCN': lambda d: GCN(d.num_features, 128, d.num_classes),
        'GAT': lambda d: GAT(d.num_features, 64, d.num_classes),
        'GraphSAGE': lambda d: GraphSAGE(d.num_features, 128, d.num_classes)
    }

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

        test_indices = data.test_mask.nonzero().squeeze()
        ft_indices = test_indices[:int(0.6*len(test_indices))]
        eval_indices = test_indices[int(0.6*len(test_indices)):]

        ft_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        eval_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        ft_mask[ft_indices] = True
        eval_mask[eval_indices] = True

        for model_name, model_fn in models.items():
            generator = TriggerGenerator(dataset.num_features, 64, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
            wm_model = model_fn(dataset)
            bi_level_optimization(wm_model, generator, data)

            optimizer = torch.optim.Adam(wm_model.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()

            wm_accuracies = []

            for epoch in range(101):
                trigger_data = generate_trigger_graph(data, generator, wm_model)

                pad_len = trigger_data.num_nodes - ft_mask.size(0)
                padded_ft_mask = ft_mask
                if pad_len > 0:
                    padded_ft_mask = torch.cat([
                        ft_mask,
                        torch.zeros(pad_len, dtype=torch.bool, device=ft_mask.device)
                    ])

                wm_model.train()
                optimizer.zero_grad()
                out = wm_model(trigger_data.x, trigger_data.edge_index)
                loss = criterion(out[padded_ft_mask], trigger_data.y[padded_ft_mask])
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    wm_model.eval()
                    with torch.no_grad():
                        pred = wm_model(trigger_data.x, trigger_data.edge_index).argmax(dim=1)
                        if hasattr(trigger_data, 'trigger_nodes'):
                            wm_mask = torch.zeros(trigger_data.num_nodes, dtype=torch.bool)
                            wm_mask[trigger_data.trigger_nodes] = True
                            wm_acc = (pred[wm_mask] == trigger_data.y[wm_mask]).float().mean().item() * 100
                            wm_accuracies.append(wm_acc)

            results[model_name][dataset_name] = wm_accuracies

    # Print summary table
    print("\nFinal watermark accuracy after fine-tuning (%):")
    print("| Dataset |  GCN   |  GAT   | GraphSAGE |")
    print("|---------|--------|--------|-----------|")
    for dataset in datasets:
        gcn_acc = results['GCN'][dataset][-1]
        gat_acc = results['GAT'][dataset][-1]
        sage_acc = results['GraphSAGE'][dataset][-1]
        print(f"| {dataset:<7} | {gcn_acc:6.2f} | {gat_acc:6.2f} | {sage_acc:9.2f} |")

    # Plot with correct output format (dot lines, no heavy border)
    plt.figure(figsize=(15, 4))
    for i, model_name in enumerate(models.keys()):
        plt.subplot(1, 3, i+1)
        for dataset in datasets:
            plt.plot(
                range(0, 101, 10),
                results[model_name][dataset],
                marker='o', linestyle='-', label=dataset
            )
        plt.title(f'({chr(97+i)}) {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Watermark Accuracy (%)')
        plt.ylim(90, 100)
        plt.grid(True)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig('fine_tuning_robustness.png')
    plt.show()
