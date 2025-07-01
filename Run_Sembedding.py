# run_sentence_embedding.py
import time
import rdflib
from sentence_transformers import SentenceTransformer
from modular_methods.dedup_pipeline import deduplicate_graphs, save_matches
from modular_methods.output_utils import build_final_result
start_time = time.time()
noise_levels = ['low','medium', 'high']
g1 = rdflib.Graph()

master_graph = rdflib.Graph()
g1.parse("data/healthcare_graph_Main.ttl")

#g2.parse("data/prog_data/healthcare_graph_progdups.ttl")
master_graph.parse("data/master_data.ttl")
phkg_graph = g1 + master_graph

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
for noise_level in noise_levels:
    start_time = time.time()
    g2 = rdflib.Graph()
    g2.parse(f"data/healthcare_graph_replaced_{noise_level}.ttl")
    matches = deduplicate_graphs(
        phkg_graph=phkg_graph,
        skg_graph=g2,
        embedding_model=model,
        use_hybrid=False,
        threshold=0.6,
        top_k=5,
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

    save_matches(final_result, f"matches_{noise_level}/SentenceEmbedding_top_k5.json")

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    with open("runtimes.txt", "w") as f:
        f.write(f"Total runtime: {runtime:.2f} seconds with noise being {noise_level}\n")
        

