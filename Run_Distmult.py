import rdflib
from sentence_transformers import SentenceTransformer
from modular_methods.embedding_utils import get_graph_embeddings_PyKEEN
from modular_methods.dedup_pipeline import deduplicate_graphs, save_matches
from modular_methods.output_utils import build_final_result

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

# --- Graph embeddings (DistMult)
print("Computing graph embeddings using DistMult...")
combined_graph = phkg_graph + g2
graph_embeddings = get_graph_embeddings_PyKEEN(combined_graph, model ="DistMult", dimensions=384, num_epochs=60)

# --- Deduplicate
matches = deduplicate_graphs(
    phkg_graph=phkg_graph,
    skg_graph=g2,
    embedding_model=model,
    graph_embeddings=graph_embeddings,
    use_hybrid=True,
    alpha=0.5,
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

# --- Save as JSON
save_matches(final_result, "matches/HybridDistMult_filtered.json")
