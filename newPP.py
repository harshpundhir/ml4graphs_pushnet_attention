import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
import wandb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Load CORA dataset
dataset = Planetoid(root='/tmp/Cora', name='CiteSeer')
data = dataset[0]

# Parameters
alpha = 0.5  # Restart probability
epsilon = 0.5  # Convergence threshold 1e-4
num_nodes = data.num_nodes
print(f"Number of nodes: {num_nodes}") # 2708
edge_index = data.edge_index
num_features = dataset.num_node_features
num_classes = dataset.num_classes

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="cora1",
    
#     # track hyperparameters and run metadata
#     config={
#     "alpha": alpha,
#     "epsilon": epsilon,
#     "dataset": "Cora",
#     "epochs": 200,
#     }
# )
#breakpoint()
# LPMP function
def lpmp(edge_index, num_nodes, alpha, epsilon, features):
    H = torch.zeros(num_nodes, features.size(1), device=features.device)
    deg = degree(edge_index[0], num_nodes, dtype=torch.float)

    for k in range(num_nodes):
        phi = torch.zeros(num_nodes, device=features.device)
        phi[k] = 1.0
        tes = 0
        while phi.abs().max() > epsilon:
            tes += 1
            hi = H[k] + alpha * phi[k] * features[k]
            H[k] = hi
            phi_neigh = (1 - alpha) * phi[k] / deg[edge_index[1][edge_index[0] == k]]
            phi[edge_index[1][edge_index[0] == k]] += phi_neigh
            phi[k] = 0
            print(f'working..at iteration {tes} on node {k}')
    return H

# Classifier model
class Classifier(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

# Initialize the classifier
classifier = Classifier(num_features, num_classes)

# Training and evaluation code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
classifier = classifier.to(device)

# Initial feature matrix
H_0 = data.x.to(device)

# Run LPMP to get the aggregated feature matrix H
H = lpmp(edge_index.to(device), num_nodes, alpha, epsilon, H_0)

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# Training
classifier.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = classifier(H)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    classifier.eval()
    _, pred = classifier(H).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / int(data.test_mask.sum())
    
    # Log metrics with wandb
    #wandb.log({"loss": loss.item(), "accuracy": accuracy, "epoch": epoch})
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{200}, Loss: {loss.item()}')

# Evaluation
classifier.eval()
_, pred = classifier(H).max(dim=1)
y_true = data.y[data.test_mask].cpu().numpy()
y_pred = pred[data.test_mask].cpu().numpy()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
conf_matrix = confusion_matrix(y_true, y_pred)
# wandb.log({
#     "final_accuracy": accuracy,
#     "precision": precision,
#     "recall": recall,
#     "f1_score": f1_score,
#     "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=[str(i) for i in range(num_classes)])
# })
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")