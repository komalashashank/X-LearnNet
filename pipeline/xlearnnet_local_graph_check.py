import runpy
import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph

print("Loading original graph and trained model...")

# Reuse the existing explanation script's graph construction and model loading.
# Prevent its GNNExplainer section from running by executing only the
# required setup through the prediction section.
source = open("pipeline/xlearnnet_graphsage_explain.py").read()

marker = "# GNN EXPLAINER"
if marker not in source:
    raise RuntimeError("Could not locate GNN EXPLAINER section.")

setup = source.split(marker)[0]
namespace = {"__name__": "__main__"}
exec(compile(setup, "pipeline/xlearnnet_graphsage_explain.py", "exec"), namespace)

data = namespace["data"]
model = namespace["model"]
student_id = namespace["student_id"]
student_idx = namespace["student_idx"]

model.eval()

with torch.no_grad():
    full_logit = model(data.x, data.edge_index)[student_idx]
    full_probability = torch.sigmoid(full_logit).item()

# Extract the two-hop computation subgraph.
subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=student_idx,
    num_hops=2,
    edge_index=data.edge_index,
    relabel_nodes=True,
    num_nodes=data.num_nodes,
    flow="source_to_target"
)

local_x = data.x[subset]
local_student_idx = int(mapping.item())

with torch.no_grad():
    local_logit = model(local_x, sub_edge_index)[local_student_idx]
    local_probability = torch.sigmoid(local_logit).item()

print("\n===== TWO-HOP SUBGRAPH VERIFICATION =====")
print("Student:", student_id)
print("Original nodes:", data.num_nodes)
print("Local nodes:", len(subset))
print("Original edges:", data.edge_index.shape[1])
print("Local edges:", sub_edge_index.shape[1])
print("Full graph probability:", round(full_probability, 8))
print("Local graph probability:", round(local_probability, 8))
print("Absolute difference:", abs(full_probability - local_probability))

if abs(full_probability - local_probability) < 1e-5:
    print("\nSUCCESS: Local subgraph preserves the original prediction.")
else:
    print("\nWARNING: Predictions differ. Do not interpret the local explanation yet.")
