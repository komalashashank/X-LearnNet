import runpy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer

print("Loading original graph and model...")

source = open("pipeline/xlearnnet_graphsage_explain.py").read()
marker = "# GNN EXPLAINER"
setup = source.split(marker)[0]

namespace = {"__name__": "__main__"}
exec(compile(setup, "pipeline/xlearnnet_graphsage_explain.py", "exec"), namespace)

data = namespace["data"]
model = namespace["model"]
student_id = namespace["student_id"]
student_idx = namespace["student_idx"]
edge_types = namespace["edge_types"]
node_ids = namespace["node_ids"]
feature_names = ['age', 'GPA', 'course_load', 'avg_course_grade', 'attendance_rate', 'lms_logins_past_month', 'avg_session_duration_minutes', 'assignment_submission_rate', 'forum_participation_count', 'video_completion_rate', 'cohort_year', 'prior_gpa', 'average_assessment_score', 'assessment_count', 'total_classes', 'attendance_percentage', 'total_enrollments', 'total_lms_events', 'average_activity_duration', 'average_weekly_study_hours', 'gender_encoded_Male', 'gender_encoded_Female', 'gender_encoded_Other', 'gender_encoded_Not reported', 'gender_encoded___unknown', 'major_encoded_Arts', 'major_encoded_Computer Science', 'major_encoded_Engineering', 'major_encoded_Business', 'major_encoded_Undeclared', 'major_encoded___unknown', 'semester_encoded_2026-Spring', 'semester_encoded_2026-Fall', 'semester_encoded_2025-Fall', 'semester_encoded___unknown', 'program_level_encoded_Undergraduate', 'program_level_encoded_Postgraduate', 'program_level_encoded___unknown']

model.eval()

# Original full-graph prediction
with torch.no_grad():
    full_probability = torch.sigmoid(
        model(data.x, data.edge_index)[student_idx]
    ).item()

# Extract two-hop computation graph
subset, local_edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=student_idx,
    num_hops=2,
    edge_index=data.edge_index,
    relabel_nodes=True,
    num_nodes=data.num_nodes,
    flow="source_to_target"
)

local_x = data.x[subset]
local_student_idx = int(mapping.item())

local_data = type(data)(
    x=local_x,
    edge_index=local_edge_index
)

with torch.no_grad():
    local_probability = torch.sigmoid(
        model(local_data.x, local_data.edge_index)[local_student_idx]
    ).item()

print("\nFull graph probability:", full_probability)
print("Local graph probability:", local_probability)

if abs(full_probability - local_probability) >= 1e-5:
    raise RuntimeError("Local prediction does not match full graph.")

print("Prediction verification passed.")

# Explain only the selected student's prediction
print("\nRunning local GNNExplainer...")

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
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
    x=local_data.x,
    edge_index=local_data.edge_index,
    index=local_student_idx
)

# Extract feature mask for selected student
node_mask = explanation.node_mask.detach().cpu()

if node_mask.ndim == 2 and node_mask.shape[0] == local_data.num_nodes:
    feature_scores = node_mask[local_student_idx].numpy()
elif node_mask.ndim == 2:
    feature_scores = node_mask[0].numpy()
else:
    feature_scores = node_mask.numpy()

if len(feature_scores) != len(feature_names):
    raise ValueError(
        f"Feature count mismatch: {len(feature_scores)} vs {len(feature_names)}"
    )

feature_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_scores
}).sort_values("importance", ascending=False)

out_dir = "outputs/graphsage"
import os
os.makedirs(out_dir, exist_ok=True)

feature_df.to_csv(
    f"{out_dir}/local_gnn_feature_explanations.csv",
    index=False
)

# Extract edge mask
edge_mask_scores = explanation.edge_mask.detach().cpu().numpy()

edge_df = pd.DataFrame({
    "source_local": local_edge_index[0].cpu().numpy(),
    "target_local": local_edge_index[1].cpu().numpy(),
    "importance": edge_mask_scores
})

edge_df["source_node"] = [
    node_ids[int(subset[i])] for i in edge_df["source_local"]
]
edge_df["target_node"] = [
    node_ids[int(subset[i])] for i in edge_df["target_local"]
]

edge_df = edge_df.sort_values("importance", ascending=False)

edge_df.to_csv(
    f"{out_dir}/local_gnn_edge_explanations.csv",
    index=False
)

# Top feature chart
top_features = feature_df.head(15).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("GNNExplainer mask importance")
plt.title(f"Local GraphSAGE Explanation: {student_id}")
plt.tight_layout()
plt.savefig(f"{out_dir}/local_gnn_feature_explanations.png", dpi=300)
plt.close()

print("\n===== TOP 15 FEATURES =====")
print(feature_df.head(15).to_string(index=False))

print("\n===== TOP 20 EDGES =====")
print(edge_df.head(20)[
    ["source_node", "target_node", "importance"]
].to_string(index=False))

print("\nSaved:")
print(f"{out_dir}/local_gnn_feature_explanations.csv")
print(f"{out_dir}/local_gnn_edge_explanations.csv")
print(f"{out_dir}/local_gnn_feature_explanations.png")
