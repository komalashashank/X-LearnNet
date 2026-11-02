
import pandas as pd

edges = pd.read_csv("data/raw/knowledge_graph_edges.csv")

print("KNOWLEDGE GRAPH ANALYSIS")
print("=" * 50)

# Count edges by relationship type
print("\nRelationship Distribution:")
print(edges["edge_type"].value_counts())

# Identify duplicate relationships
print("\nDuplicate Edges by Type:")

duplicates = edges[
    edges.duplicated(
        subset=["source_node_id", "edge_type", "target_node_id"],
        keep=False
    )
]

print(duplicates["edge_type"].value_counts())

# Count unique relationships
unique_edges = edges.drop_duplicates(
    subset=["source_node_id", "edge_type", "target_node_id"]
)

print("\nTotal edges:", len(edges))
print("Unique edges:", len(unique_edges))
print("Duplicate edges:", len(edges) - len(unique_edges))

# Save analysis
duplicates.to_csv(
    "data/output/duplicate_graph_edges.csv",
    index=False
)

unique_edges.to_csv(
    "data/output/unique_graph_edges.csv",
    index=False
)

print("\nGraph analysis completed!")
