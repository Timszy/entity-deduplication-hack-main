# run_sentence_embedding.py

import rdflib
from sentence_transformers import SentenceTransformer
from modular_methods.dedup_pipeline import deduplicate_graphs, save_matches
from modular_methods.output_utils import build_final_result


g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()
g1.parse("data/healthcare_graph_original_v2.ttl")
g2.parse("data/prog_data/healthcare_graph_progdups.ttl")
master_graph.parse("data/master_data.ttl")
phkg_graph = g1 + master_graph

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

matches = deduplicate_graphs(
    phkg_graph=phkg_graph,
    skg_graph=g2,
    embedding_model=model,
    use_hybrid=False,
    threshold=0.6,
    top_k=2,
    filter_literals=True,
)

print(f"Found {len(matches)} filtered matches.")

final_result = build_final_result(
    matches,
    phkg_graph,  # or your first graph
    g2,          # or your second graph
    graph1_name="phkg_graph",
    graph2_name="g2"
)

save_matches(final_result, "matches/SentenceEmbedding_filterednew.json")

