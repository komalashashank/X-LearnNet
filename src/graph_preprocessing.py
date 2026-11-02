
import pandas as pd

# Load original graph
edges = pd.read_csv(
    "data/raw/knowledge_graph_edges.csv"
)

print("Original graph edges:", len(edges))

# Group identical relationships and count their frequency
weighted_edges = (
    edges.groupby(
        ["source_node_id", "edge_type", "target_node_id"],
        as_index=False
    )
    .size()
    .rename(columns={"size": "edge_weight"})
)

# Save weighted graph
weighted_edges.to_csv(
    "data/processed/knowledge_graph_weighted.csv",
    index=False
)

# Create a simple deduplicated graph
unique_edges = weighted_edges[
    ["source_node_id", "edge_type", "target_node_id"]
]

unique_edges.to_csv(
    "data/processed/knowledge_graph_deduplicated.csv",
    index=False
)

print("\nGRAPH PREPROCESSING RESULTS")
print("--------------------------------")

print("Original edges:", len(edges))
print("Unique edges:", len(unique_edges))
print("Removed duplicate rows:", len(edges) - len(unique_edges))

print("\nRelationship weights:")
print(weighted_edges.groupby("edge_type")["edge_weight"].sum())

print("\nGraph preprocessing completed!")
