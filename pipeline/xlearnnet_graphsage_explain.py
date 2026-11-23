import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE = "hdfs://localhost:9000/xlearnnet"
TARGET = "course_failure_flag"

OUTPUT_DIR = "outputs/graphsage"
MODEL_PATH = os.path.join(OUTPUT_DIR, "graphsage_course_failure.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# GRAPH SAGE MODEL - SAME ARCHITECTURE AS TRAINING
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD GRAPH AND FEATURE DATA
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("XLearnNet-GraphSAGE-Explainability")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("Loading graph nodes and edges...")

nodes_df = spark.read.csv(
    BASE + "/raw/knowledge_graph_nodes.csv",
    header=True,
    inferSchema=True
).toPandas()

edges_df = spark.read.csv(
    BASE + "/raw/knowledge_graph_edges.csv",
    header=True,
    inferSchema=True
).toPandas()

nodes_df["node_id"] = nodes_df["node_id"].astype(str)
edges_df["source_node_id"] = edges_df["source_node_id"].astype(str)
edges_df["target_node_id"] = edges_df["target_node_id"].astype(str)

print("Loading student feature splits...")

feature_path = BASE + "/feature_engineered/" + TARGET

train_df = spark.read.parquet(feature_path + "/train").toPandas()
val_df = spark.read.parquet(feature_path + "/validation").toPandas()
test_df = spark.read.parquet(feature_path + "/test").toPandas()

student_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

student_df["student_id"] = student_df["student_id"].astype(str)

# --------------------------------------------------
# RECONSTRUCT NODE FEATURES AND GRAPH
# --------------------------------------------------

node_ids = nodes_df["node_id"].tolist()
node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

num_nodes = len(node_ids)
feature_dim = len(student_df.iloc[0]["features"])

x = np.zeros((num_nodes, feature_dim), dtype=np.float32)

for row in student_df.itertuples(index=False):
    idx = node_to_idx.get(row.student_id)
    if idx is not None:
        x[idx] = np.asarray(row.features, dtype=np.float32)

source = edges_df["source_node_id"].map(node_to_idx)
target = edges_df["target_node_id"].map(node_to_idx)

valid = source.notna() & target.notna()

src = source[valid].astype(int).to_numpy()
dst = target[valid].astype(int).to_numpy()

valid_edge_types = edges_df.loc[valid, "edge_type"].astype(str).to_numpy()

edge_index = torch.tensor(
    np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ]),
    dtype=torch.long
)

edge_types = np.concatenate([
    valid_edge_types,
    valid_edge_types
])

data = Data(
    x=torch.tensor(x, dtype=torch.float32),
    edge_index=edge_index
)

print("Graph nodes:", num_nodes)
print("Graph edges for message passing:", edge_index.shape[1])
print("Feature dimensions:", feature_dim)

# --------------------------------------------------
# SELECT A HIGH-RISK TEST STUDENT
# --------------------------------------------------

comparison = pd.read_csv(
    "outputs/graphsage_vs_logistic_comparison.csv"
)

selected = comparison.sort_values(
    "failure_probability",
    ascending=False
).iloc[0]

student_id = str(selected["student_id"])
student_idx = node_to_idx.get(student_id)

if student_idx is None:
    raise ValueError(f"Student {student_id} not found in graph nodes.")

actual = int(selected["actual"])
print("\nSelected student:", student_id)
print("Actual outcome:", actual)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

device = torch.device("cpu")

model = GraphSAGE(feature_dim).to(device)

state = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state)
model.eval()

data = data.to(device)

with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probabilities = torch.sigmoid(logits)
    probability = float(probabilities[student_idx].item())

print("GraphSAGE failure probability:", round(probability, 4))
print("Actual outcome:", actual)

# --------------------------------------------------
# GNN EXPLAINER
# --------------------------------------------------

print("\nRunning GNNExplainer...")

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=100),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config={
        "mode": "binary_classification",
        "task_level": "node",
        "return_type": "raw"
    }
)

explanation = explainer(
    x=data.x,
    edge_index=data.edge_index,
    index=student_idx
)

print("Explanation completed.")

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

feature_names = ['age', 'GPA', 'course_load', 'avg_course_grade', 'attendance_rate', 'lms_logins_past_month', 'avg_session_duration_minutes', 'assignment_submission_rate', 'forum_participation_count', 'video_completion_rate', 'cohort_year', 'prior_gpa', 'average_assessment_score', 'assessment_count', 'total_classes', 'attendance_percentage', 'total_enrollments', 'total_lms_events', 'average_activity_duration', 'average_weekly_study_hours', 'gender_encoded_Male', 'gender_encoded_Female', 'gender_encoded_Other', 'gender_encoded_Not reported', 'gender_encoded___unknown', 'major_encoded_Arts', 'major_encoded_Computer Science', 'major_encoded_Engineering', 'major_encoded_Business', 'major_encoded_Undeclared', 'major_encoded___unknown', 'semester_encoded_2026-Spring', 'semester_encoded_2026-Fall', 'semester_encoded_2025-Fall', 'semester_encoded___unknown', 'program_level_encoded_Undergraduate', 'program_level_encoded_Postgraduate', 'program_level_encoded___unknown']

node_mask = explanation.node_mask.detach().cpu().numpy()

if node_mask.ndim == 2:
    if node_mask.shape[0] == num_nodes:
        feature_scores = node_mask[student_idx]
    else:
        feature_scores = node_mask[0]
else:
    feature_scores = node_mask.reshape(-1)

assert len(feature_scores) == len(feature_names), (
    f"Feature mask mismatch: {len(feature_scores)} vs {len(feature_names)}"
)

feature_ranking = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_scores
}).sort_values("importance", ascending=False)

print("\nTOP 15 GRAPH FEATURE IMPORTANCES")
print(feature_ranking.head(15).to_string(index=False))

feature_ranking.to_csv(
    os.path.join(OUTPUT_DIR, "graphsage_feature_explanations.csv"),
    index=False
)

# Feature importance chart
top_features = feature_ranking.head(15).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("GNNExplainer importance")
plt.title(f"GraphSAGE Feature Explanation - {student_id}")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "graphsage_feature_explanation.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# EDGE IMPORTANCE
# --------------------------------------------------

edge_mask = explanation.edge_mask.detach().cpu().numpy()

top_indices = np.argsort(edge_mask)[::-1][:20]

edge_rows = []

for edge_position in top_indices:
    source_idx = int(edge_index[0, edge_position].item())
    target_idx = int(edge_index[1, edge_position].item())

    edge_rows.append({
        "source_node_id": node_ids[source_idx],
        "source_node_type": nodes_df.iloc[source_idx]["node_type"],
        "target_node_id": node_ids[target_idx],
        "target_node_type": nodes_df.iloc[target_idx]["node_type"],
        "edge_type": edge_types[edge_position],
        "importance": float(edge_mask[edge_position])
    })

edge_ranking = pd.DataFrame(edge_rows)

print("\nTOP GRAPH CONNECTIONS")
print(edge_ranking.head(15).to_string(index=False))

edge_ranking.to_csv(
    os.path.join(OUTPUT_DIR, "graphsage_edge_explanations.csv"),
    index=False
)

# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "graphsage_explanation_summary.txt"
)

with open(summary_path, "w") as f:
    f.write(f"Student ID: {student_id}\n")
    f.write(f"Actual outcome: {actual}\n")
    f.write(f"GraphSAGE failure probability: {probability:.6f}\n")
    f.write("\nTop 10 feature explanations:\n")

    for row in feature_ranking.head(10).itertuples(index=False):
        f.write(f"{row.feature}: {row.importance:.6f}\n")

    f.write("\nTop 10 graph connections:\n")

    for row in edge_ranking.head(10).itertuples(index=False):
        f.write(
            f"{row.source_node_id} -> {row.target_node_id} "
            f"({row.edge_type}): {row.importance:.6f}\n"
        )

print("\nGraphSAGE explanations saved in:", OUTPUT_DIR)
print("Summary:", summary_path)

spark.stop()
