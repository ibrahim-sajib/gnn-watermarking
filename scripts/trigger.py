import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy
from torch_geometric.utils import to_dense_adj, dense_to_sparse





class TriggerGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, owner_id):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)
        self.owner_id = owner_id

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = torch.sigmoid(self.conv2(x, edge_index))
        out = x.clone()
        out[:, -5:] = self.owner_id
        return out




def generate_trigger_graph(data, generator, target_model, num_triggers=50):
    with torch.no_grad():
        probs = F.softmax(target_model(data.x, data.edge_index), dim=1)

    selected_nodes = []
    for class_idx in range(probs.size(1)):
        class_nodes = torch.where(data.y == class_idx)[0]
        if len(class_nodes) > 0:
            selected_nodes.append(class_nodes[probs[class_nodes, class_idx].argmax()].item())

    trigger_features = generator(data.x, data.edge_index)
    trigger_nodes = list(range(data.num_nodes, data.num_nodes + num_triggers))

    adj = to_dense_adj(data.edge_index)[0]
    new_adj = torch.zeros((adj.size(0) + num_triggers, adj.size(1) + num_triggers), device=adj.device)
    new_adj[:adj.size(0), :adj.size(1)] = adj

    for i, trigger in enumerate(trigger_nodes):
        for node in selected_nodes:
            new_adj[node, trigger] = 1
            new_adj[trigger, node] = 1

    new_data = copy.deepcopy(data)
    new_data.x = torch.cat([data.x, trigger_features[:num_triggers]], dim=0)
    new_data.edge_index = dense_to_sparse(new_adj)[0]
    new_data.y = torch.cat([data.y, torch.zeros(num_triggers, dtype=torch.long, device=data.y.device)])

    # Preserve original test mask before adding triggers
    new_data.original_test_mask = data.test_mask.clone()

    # Add trigger information
    new_data.trigger_nodes = trigger_nodes
    new_data.selected_nodes = selected_nodes
    new_data.trigger_mask = torch.zeros(new_data.num_nodes, dtype=torch.bool, device=data.x.device)
    new_data.trigger_mask[trigger_nodes] = True

    return new_data