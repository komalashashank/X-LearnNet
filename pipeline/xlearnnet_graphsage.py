import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pyspark.sql import SparkSession
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

# ---------------- CONFIGURATION ----------------

TARGET = "course_failure_flag"
BASE = "hdfs://localhost:9000/xlearnnet"
GRAPH_NODES = BASE + "/raw/knowledge_graph_nodes.csv"
GRAPH_EDGES = BASE + "/raw/knowledge_graph_edges.csv"
FEATURE_PATH = BASE + "/feature_engineered/" + TARGET

OUTPUT_DIR = "outputs/graphsage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ---------------- SPARK ----------------

spark = (
    SparkSession.builder
    .appName("XLearnNet-GraphSAGE")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("Loading graph nodes and edges...")

nodes_df = spark.read.csv(
    GRAPH_NODES, header=True, inferSchema=True
).toPandas()

edges_df = spark.read.csv(
    GRAPH_EDGES, header=True, inferSchema=True
).toPandas()

print("Nodes:", len(nodes_df))
print("Edges:", len(edges_df))
print("Node types:")
print(nodes_df["node_type"].value_counts())

print("Loading feature-engineered student data...")

train_df = spark.read.parquet(FEATURE_PATH + "/train").toPandas()
val_df = spark.read.parquet(FEATURE_PATH + "/validation").toPandas()
test_df = spark.read.parquet(FEATURE_PATH + "/test").toPandas()

student_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print("Student records:", len(student_df))

# ---------------- NODE MAPPING ----------------

nodes_df["node_id"] = nodes_df["node_id"].astype(str)
edges_df["source_node_id"] = edges_df["source_node_id"].astype(str)
edges_df["target_node_id"] = edges_df["target_node_id"].astype(str)
student_df["student_id"] = student_df["student_id"].astype(str)

node_ids = nodes_df["node_id"].tolist()
node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

num_nodes = len(node_ids)
student_mask = nodes_df["node_type"].eq("Student").to_numpy()

print("Total graph nodes:", num_nodes)
print("Student nodes:", int(student_mask.sum()))

# ---------------- NODE FEATURES ----------------

sample_vector = student_df["features"].iloc[0]
feature_dim = len(sample_vector)

x = np.zeros((num_nodes, feature_dim), dtype=np.float32)

student_index = {}
for row in student_df.itertuples(index=False):
    student_index[row.student_id] = row

for student_id, row in student_index.items():
    idx = node_to_idx.get(student_id)
    if idx is not None:
        x[idx] = np.asarray(row.features, dtype=np.float32)

# ---------------- GRAPH EDGES ----------------

print("Building graph connections...")

source = edges_df["source_node_id"].map(node_to_idx)
target = edges_df["target_node_id"].map(node_to_idx)

valid = source.notna() & target.notna()

src = source[valid].astype(int).to_numpy()
dst = target[valid].astype(int).to_numpy()

# Make relationships bidirectional for message passing
edge_index = torch.tensor(
    np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ]),
    dtype=torch.long
)

# ---------------- LABELS AND SPLITS ----------------

y = np.zeros(num_nodes, dtype=np.float32)

train_mask = np.zeros(num_nodes, dtype=bool)
val_mask = np.zeros(num_nodes, dtype=bool)
test_mask = np.zeros(num_nodes, dtype=bool)

for row in student_df.itertuples(index=False):
    idx = node_to_idx.get(row.student_id)
    if idx is None:
        continue

    y[idx] = float(row.label)

    if row.student_id in set(train_df["student_id"].astype(str)):
        train_mask[idx] = True
    elif row.student_id in set(val_df["student_id"].astype(str)):
        val_mask[idx] = True
    elif row.student_id in set(test_df["student_id"].astype(str)):
        test_mask[idx] = True

data = Data(
    x=torch.tensor(x, dtype=torch.float32),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.float32)
)

train_mask = torch.tensor(train_mask)
val_mask = torch.tensor(val_mask)
test_mask = torch.tensor(test_mask)

print("Train students:", int(train_mask.sum()))
print("Validation students:", int(val_mask.sum()))
print("Test students:", int(test_mask.sum()))
print("Feature dimensions:", feature_dim)
print("Graph edges for message passing:", edge_index.shape[1])

# ---------------- GRAPHSAGE MODEL ----------------

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.classifier(x).squeeze(-1)


device = torch.device("cpu")
model = GraphSAGE(feature_dim).to(device)

data = data.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

positive = data.y[train_mask].sum()
negative = train_mask.sum() - positive
pos_weight = negative / positive.clamp(min=1)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# ---------------- TRAINING ----------------

print("\nTraining GraphSAGE...")

best_val_f1 = -1
best_state = None
best_threshold = 0.5
patience = 8
wait = 0

for epoch in range(1, 41):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = criterion(logits[train_mask], data.y[train_mask])

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(
            model(data.x, data.edge_index)[val_mask]
        ).cpu().numpy()

    val_true = data.y[val_mask].cpu().numpy()

    thresholds = np.arange(0.20, 0.81, 0.02)
    scores = [
        f1_score(val_true, (val_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]

    best_idx = int(np.argmax(scores))
    epoch_f1 = scores[best_idx]

    print(
        f"Epoch {epoch:02d} | Loss: {loss.item():.4f} "
        f"| Validation F1: {epoch_f1:.4f}"
    )

    if epoch_f1 > best_val_f1:
        best_val_f1 = epoch_f1
        best_threshold = float(thresholds[best_idx])
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        print("Early stopping.")
        break

# ---------------- FINAL TEST EVALUATION ----------------

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    all_prob = torch.sigmoid(
        model(data.x, data.edge_index)
    ).cpu().numpy()

test_true = data.y[test_mask].cpu().numpy()
test_prob = all_prob[test_mask.cpu().numpy()]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n========== GRAPHSAGE TEST RESULTS ==========")
print("Selected validation threshold:", best_threshold)
print("Accuracy:", round(accuracy_score(test_true, test_pred), 4))
print("Precision:", round(precision_score(test_true, test_pred, zero_division=0), 4))
print("Recall:", round(recall_score(test_true, test_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(test_true, test_pred, zero_division=0), 4))
print("Confusion matrix:\n", confusion_matrix(test_true, test_pred))

if len(np.unique(test_true)) == 2:
    print("ROC-AUC:", round(roc_auc_score(test_true, test_prob), 4))
    print("PR-AUC:", round(average_precision_score(test_true, test_prob), 4))

print("\nClassification report:")
print(classification_report(test_true, test_pred, zero_division=0))

# ---------------- SAVE RESULTS ----------------

torch.save(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, "graphsage_course_failure.pt")
)

results = pd.DataFrame({
    "actual": test_true.astype(int),
    "predicted": test_pred,
    "failure_probability": test_prob
})

results.to_csv(
    os.path.join(OUTPUT_DIR, "graphsage_test_predictions.csv"),
    index=False
)

with open(os.path.join(OUTPUT_DIR, "graphsage_metrics.txt"), "w") as f:
    f.write(f"Validation threshold: {best_threshold}\n")
    f.write(f"Validation F1: {best_val_f1}\n")
    f.write(f"Test accuracy: {accuracy_score(test_true, test_pred)}\n")
    f.write(f"Test precision: {precision_score(test_true, test_pred, zero_division=0)}\n")
    f.write(f"Test recall: {recall_score(test_true, test_pred, zero_division=0)}\n")
    f.write(f"Test F1: {f1_score(test_true, test_pred, zero_division=0)}\n")

print("\nGraphSAGE model and results saved in:", OUTPUT_DIR)

spark.stop()
