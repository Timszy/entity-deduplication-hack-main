import rdflib
from sentence_transformers import SentenceTransformer
from modular_methods.embedding_utils import get_graph_embeddings_Node2vec
from modular_methods.dedup_pipeline import deduplicate_graphs, save_matches
from modular_methods.output_utils import build_final_result  
import time

start_time = time.time()

# --- Load RDF graphs
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()
g1.parse("data/healthcare_graph_original_v2.ttl")
g2.parse("data/prog_data/healthcare_graph_progdups.ttl")
master_graph.parse("data/master_data.ttl")
phkg_graph = g1 + master_graph

# --- Sentence embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- Graph embeddings (Node2Vec)
print("Computing graph embeddings...")
combined_graph = phkg_graph + g2
graph_embeddings = get_graph_embeddings_Node2vec(combined_graph, dimensions=384)
# --- Run deduplication for multiple alpha values
alpha_values = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]  # Change as needed
for alpha in alpha_values:
    matches = deduplicate_graphs(
        phkg_graph=phkg_graph,
        skg_graph=g2,
        embedding_model=model,
        graph_embeddings=graph_embeddings,
        use_hybrid=True,
        alpha=alpha,
        text_dim=384,
        threshold=0.6,
        top_k=2,
        filter_literals=True,
    )
    print(f"Found {len(matches)} filtered matches.")

    # --- Format result for output
    final_result = build_final_result(
        matches,
        phkg_graph,
        g2,
        graph1_name="phkg_graph",
        graph2_name="g2"
    )

    output_path = f"matches/HybridNode2Vec_alpha_{alpha}.json"
    save_matches(final_result, output_path)
    print(f"Saved matches to {output_path}")
end_time = time.time()
runtime = end_time - start_time
print(f"Total runtime: {runtime:.2f} seconds")

# Save runtime to file
with open("runtimes.txt", "a") as f:
    f.write(f"Run with model = Node2Vec and alpha={alpha_values} took {runtime:.2f} seconds\n")