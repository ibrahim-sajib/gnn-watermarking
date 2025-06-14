import torch
import torch.nn.functional as F
import random
import copy
from sklearn.metrics import roc_curve, auc
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
import os
os.makedirs("results", exist_ok=True)

from scripts.trigger import TriggerGenerator, generate_trigger_graph
from scripts.train import bi_level_optimization
from models_def.gcn import GCN
from scripts.train import train_model
from scripts.metrics import load_dataset



def run_evasion_attack():
    datasets = ['Cora', 'Pubmed']
    models = {
        'GCN': lambda d: GCN(d.num_features, 128, d.num_classes),
    }

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (area = 0.5)')

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        data = dataset[0]

        # Our Method
        generator = TriggerGenerator(dataset.num_features, 64, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
        model = models['GCN'](dataset)
        bi_level_optimization(model, generator, data)
        trigger_data = generate_trigger_graph(data, generator, model)

        cosine_sims = []
        is_trigger = []
        for i in range(trigger_data.num_nodes):
            if i in trigger_data.trigger_nodes:
                is_trigger.append(1)
                neighbors = trigger_data.edge_index[1][trigger_data.edge_index[0] == i]
                if len(neighbors) > 0:
                    neighbor_features = trigger_data.x[neighbors]
                    sim = F.cosine_similarity(trigger_data.x[i].unsqueeze(0), neighbor_features).mean().item()
                else:
                    sim = 0.0
            else:
                is_trigger.append(0)
                neighbors = trigger_data.edge_index[1][trigger_data.edge_index[0] == i]
                if len(neighbors) > 0:
                    neighbor_features = trigger_data.x[neighbors]
                    sim = F.cosine_similarity(trigger_data.x[i].unsqueeze(0), neighbor_features).mean().item()
                else:
                    sim = 1.0
            cosine_sims.append(sim)

        y_true = np.array(is_trigger)
        y_scores = 1 - np.array(cosine_sims)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Ours on {dataset_name} (area = {roc_auc:.3f})')

        # GTA Baseline
        gta_data = copy.deepcopy(data)
        trigger_nodes = random.sample(range(data.num_nodes), 50)
        feature_dim = data.x.size(1)
        for node in trigger_nodes:
            gta_data.x[node] = torch.zeros_like(gta_data.x[node])
            gta_data.x[node, :5] = 5.0
            gta_data.y[node] = 0

        adj = to_dense_adj(gta_data.edge_index)[0]
        for i in trigger_nodes:
            for j in trigger_nodes:
                if i != j:
                    adj[i,j] = 1
        gta_data.edge_index = dense_to_sparse(adj)[0]

        gta_model = models['GCN'](dataset)
        train_model(gta_model, gta_data)

        gta_scores = []
        degrees = torch.zeros(gta_data.num_nodes)
        for i in range(gta_data.num_nodes):
            degrees[i] = (gta_data.edge_index[0] == i).sum()

        for i in range(gta_data.num_nodes):
            neighbors = gta_data.edge_index[1][gta_data.edge_index[0] == i]
            if len(neighbors) > 0:
                neighbor_feats = gta_data.x[neighbors]
                feat_consistency = 1 - F.cosine_similarity(
                    gta_data.x[i].unsqueeze(0), neighbor_feats).mean().item()
            else:
                feat_consistency = 0
            gta_scores.append(degrees[i].item() * feat_consistency)

        gta_y_true = np.array([1 if i in trigger_nodes else 0 for i in range(gta_data.num_nodes)])
        gta_fpr, gta_tpr, _ = roc_curve(gta_y_true, gta_scores)
        gta_auc = auc(gta_fpr, gta_tpr)
        plt.plot(gta_fpr, gta_tpr, '-.', label=f'GTA on {dataset_name} (area = {gta_auc:.3f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Trigger Node Detection')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('results/evasion_roc_curve.png')
    # plt.show()





if __name__ == "__main__":
    run_evasion_attack()
