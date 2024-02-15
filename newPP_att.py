import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
import wandb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch_geometric.nn import GATConv
import torch.nn as nn
import pandas as pd

# Load CORA dataset
dataset = Planetoid(root='/tmp/Cora', name='CiteSeer')
data = dataset[0]
ATTENT = True

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

# attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(in_dim, 1)  # Compute attention score for each node
        self.fc = nn.Linear(in_dim, out_dim)  # Classifier layer
    
    def forward(self, x):
        attention_scores = F.softmax(self.attention_fc(x), dim=0)  # Compute attention scores
        weighted_features = x * attention_scores  # Apply attention scores to features
        return F.log_softmax(self.fc(weighted_features), dim=1)

# attention classifier
class ClassifierWithAttention(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ClassifierWithAttention, self).__init__()
        self.attention_layer = AttentionLayer(in_dim, out_dim)
    
    def forward(self, x):
        return self.attention_layer(x)

# Classifier model
class Classifier(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

# Initialize the classifier

def evaluation_(attent, alpha, epsilon, data):
    if attent:
        classifier = ClassifierWithAttention(num_features, num_classes)
    else:
        classifier = Classifier(num_features, num_classes) # classifier_with_attention = ClassifierWithAttention(num_features, num_classes)


    # Training and evaluation code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    return accuracy, precision, recall, f1_score
    

# evaluation


aplhas = [0.5, 0.6, 0.7, 0.8, 0.9]
epsilons = [0.5, 0.6, 0.7, 0.8, 0.9]
attents = [True, False]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

results_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Alpha', 'Epsilon', 'Attention'])
for alpha in aplhas:
    for epsilon in epsilons:
        for attent in attents:
            wandb.init(project="cs",
                       config={
                           "alpha": alpha,
                           "epsilon": epsilon,
                           "attent": attent,
                           "epochs": 200,
                           "dataset": "Cora"
                       })
            accuracy, precision, recall, f1_score = evaluation_(attent, alpha, epsilon, data)
            if attent:
                attent_bin = 1
            else:
                attent_bin = 0
            current_result_df = pd.DataFrame({
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1_Score': [f1_score],
                'Alpha': [alpha],
                'Epsilon': [epsilon],
                'Attention': [attent_bin]  # Use 1 for True, 0 for False
            })
            # Concatenate the current result to the main DataFrame
            results_df = pd.concat([results_df, current_result_df], ignore_index=True)
            print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
            wandb.log({
                "final_accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "alpha": alpha,
                "epsilon": epsilon,
                "attent": attent_bin
                    })
            
results_df.to_csv('results_cs.csv')