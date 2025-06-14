import torch
import matplotlib.pyplot as plt
import os
os.makedirs("results", exist_ok=True)

from scripts.trigger import TriggerGenerator, generate_trigger_graph
from scripts.train import bi_level_optimization
from scripts.metrics import load_dataset
from models_def.gat import GAT



def run_fraudulent_declaration():
    dataset = load_dataset('Cora')
    data = dataset[0]
    indices = torch.randperm(data.num_nodes)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[indices[:int(0.6*data.num_nodes)]] = True
    data.val_mask[indices[int(0.6*data.num_nodes):int(0.8*data.num_nodes)]] = True
    data.test_mask[indices[int(0.8*data.num_nodes):]] = True

    # Create 5 watermarked models with different owner IDs
    models = []
    generators = []
    owner_ids = [
        torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]),
        torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1]),
        torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]),
        torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]),
        torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    ]

    for owner_id in owner_ids:
        generator = TriggerGenerator(dataset.num_features, 64, owner_id)
        model = GAT(dataset.num_features, 64, dataset.num_classes)
        bi_level_optimization(model, generator, data)
        models.append(model)
        generators.append(generator)

    # Create verification matrix
    verification_matrix = torch.zeros((5, 5))
    for i, gen in enumerate(generators):
        for j, model in enumerate(models):
            trigger_data = generate_trigger_graph(data, gen, model)
            with torch.no_grad():
                pred = model(trigger_data.x, trigger_data.edge_index).argmax(dim=1)
                wm_mask = torch.zeros(trigger_data.num_nodes, dtype=torch.bool)
                wm_mask[trigger_data.trigger_nodes] = True
                wm_acc = (pred[wm_mask] == trigger_data.y[wm_mask]).float().mean().item() * 100
                verification_matrix[i,j] = wm_acc

    # Print verification matrix
    print("\nFraudulent Declaration Test Results (% accuracy):")
    print("| Model | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5 |")
    print("|-------|-------|-------|-------|-------|-------|")
    for i in range(5):
        print(f"| Model {i+1} | {verification_matrix[0,i]:5.1f} | {verification_matrix[1,i]:5.1f} | {verification_matrix[2,i]:5.1f} | {verification_matrix[3,i]:5.1f} | {verification_matrix[4,i]:5.1f} |")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(verification_matrix, cmap='viridis', vmin=0, vmax=100)
    plt.colorbar(label='Watermark Accuracy (%)')
    plt.xticks(range(5), [f'Gen {i+1}' for i in range(5)])
    plt.yticks(range(5), [f'Model {i+1}' for i in range(5)])
    plt.title("Fraudulent Declaration Attack Results")
    plt.xlabel("Generator Used for Verification")
    plt.ylabel("Target Model")
    plt.savefig('results/fraudulent_declaration.png')
    # plt.show()




if __name__ == "__main__":
    run_fraudulent_declaration()
