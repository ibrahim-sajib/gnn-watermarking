
import torch
from torch_geometric.datasets import Planetoid
import os
import pandas as pd
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)



from models_def.gat import GAT
from models_def.gcn import GCN
from models_def.graphsage import GraphSAGE
from scripts.metrics import calculate_metrics
from scripts.trigger import TriggerGenerator, generate_trigger_graph
from scripts.train import train_model, bi_level_optimization
from scripts.metrics import load_dataset


def run_comparison():
    datasets = ['Cora', 'Pubmed']
    models = {
        'GCN': lambda d: GCN(d.num_features, 128, d.num_classes),
        'GAT': lambda d: GAT(d.num_features, 64, d.num_classes),
        'GraphSAGE': lambda d: GraphSAGE(d.num_features, 128, d.num_classes)
    }

    results = []

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
            original_model = model_fn(dataset)
            orig_metrics = train_model(original_model, data, wm_weight=0.0)
            torch.save(original_model.state_dict(), f'models/{dataset_name}_{model_name}_original.pth')

            generator = TriggerGenerator(dataset.num_features, 64, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
            wm_model = model_fn(dataset)
            bi_level_optimization(wm_model, generator, data)
            torch.save(wm_model.state_dict(), f'models/{dataset_name}_{model_name}_watermarked.pth')

            trigger_data = generate_trigger_graph(data, generator, wm_model)
            wm_metrics = calculate_metrics(wm_model, trigger_data)

            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Orig_Acc': orig_metrics['accuracy']*100,
                'WM_Acc': wm_metrics['accuracy']*100,
                'Orig_Prec': orig_metrics['precision']*100,
                'WM_Prec': wm_metrics['precision']*100,
                'Orig_Rec': orig_metrics['recall']*100,
                'WM_Rec': wm_metrics['recall']*100,
                'Orig_F1': orig_metrics['f1']*100,
                'WM_F1': wm_metrics['f1']*100
            })


    # Save results table
    df_full = pd.DataFrame(results)
    df_full.to_csv("results/model_performance_table.csv", index=False)

    # Watermark accuracy table
    wm_table = {'Dataset': [], 'GCN': [], 'GAT': [], 'GraphSAGE': []}
    for dataset in datasets:
        wm_table['Dataset'].append(dataset)
        for model in models.keys():
            res = next(r for r in results if r['Dataset'] == dataset and r['Model'] == model)
            wm_acc = res.get('WM_Acc', 0.0)
            wm_table[model].append(f"{wm_acc:.2f}")

    df_wm = pd.DataFrame(wm_table)
    df_wm.to_csv("results/watermark_accuracy_table.csv", index=False)




    # print the results
    print("\nModel performance (original model | watermarked model)")
    print("| Dataset |   Model   |  Accuracy (%)  | Precision (%) |   Recall (%)  | F1-score (%)  |")
    print("|---------|-----------|----------------|---------------|---------------|---------------|")

    for dataset in datasets:
        for i, model in enumerate(models.keys()):
            res = next(r for r in results if r['Dataset'] == dataset and r['Model'] == model)
            if i == 0:
                print(f"| {dataset:<7} | {model:<9} | {res['Orig_Acc']:.2f} | {res['WM_Acc']:.2f} | {res['Orig_Prec']:.2f} | {res['WM_Prec']:.2f} | {res['Orig_Rec']:.2f} | {res['WM_Rec']:.2f} | {res['Orig_F1']:.2f} | {res['WM_F1']:.2f} |")
            else:
                print(f"|         | {model:<9} | {res['Orig_Acc']:.2f} | {res['WM_Acc']:.2f} | {res['Orig_Prec']:.2f} | {res['WM_Prec']:.2f} | {res['Orig_Rec']:.2f} | {res['WM_Rec']:.2f} | {res['Orig_F1']:.2f} | {res['WM_F1']:.2f} |")

    wm_table = {'Dataset': [], 'GCN': [], 'GAT': [], 'GraphSAGE': []}

    for dataset in datasets:
        wm_table['Dataset'].append(dataset)
        for model in models.keys():
            res = next(r for r in results if r['Dataset'] == dataset and r['Model'] == model)
            wm_acc = res.get('WM_Acc', 0.0)
            wm_table[model].append(f"{wm_acc:.2f}")

    print("\nWatermark accuracy table (%)")
    print("| Dataset |  GCN  |  GAT  | GraphSAGE |")
    print("|---------|-------|-------|------------|")
    for i in range(len(wm_table['Dataset'])):
        print(f"| {wm_table['Dataset'][i]:<7} | {wm_table['GCN'][i]:>5} | {wm_table['GAT'][i]:>5} | {wm_table['GraphSAGE'][i]:>10} |")





if __name__ == "__main__":
    run_comparison()
