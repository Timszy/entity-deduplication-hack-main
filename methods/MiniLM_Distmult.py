import json
import pandas as pd
import rdflib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# =============================
# ========= Configs ===========
# =============================
TEXT_MODEL = "all-MiniLM-L6-v2"
TEXT_DIM = 384
THRESHOLD = 0.5
NUM_EPOCHS = 80
GRAPH_MODEL = "DistMult"
ALPHA = 0.5 # You can change this value to weight the text embedding (0.0 = is graph only)
KNOWN_PREFIXES = [
    "ucum:",
    "sphn-loinc:",
    "snomed:",
    "atc:",
    "sphn-chop:",
    "hgnc:",
    "sphn-icd-10:",
    "https://biomedit.ch/rdf/sphn-resource/ucum/",
    "https://biomedit.ch/rdf/sphn-resource/loinc/",
    "http://snomed.info/id/",
    "https://www.whocc.no/atc_ddd_index/?code=",
    "https://biomedit.ch/rdf/sphn-resource/chop/",
    "https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/",
    "https://biomedit.ch/rdf/sphn-resource/icd-10-gm/",
]

# =============================
# ==== Handling the graphs ====
# =============================
def load_graph():
    g1 = rdflib.Graph()
    g2 = rdflib.Graph()
    master_graph = rdflib.Graph()
    g1.parse("data/healthcare_graph_original_v2.ttl")
    g2.parse("data/prog_data/healthcare_graph_var_only.ttl")
    master_graph.parse("data/master_data.ttl")

    phkg_graph = g1 + master_graph
    combined_graph = phkg_graph + g2
    return combined_graph, phkg_graph, g2

def extract_triples(graph):
    return [
        (str(s), str(p), str(o))
        for s, p, o in graph
        if not isinstance(s, rdflib.BNode) and not isinstance(o, rdflib.BNode)
    ]

# =============================
# ======= Handling text =======
# =============================
def traverse_graph_and_get_literals(graph, subject) -> Dict[str, Dict[str, str]]:
    def traverse(subject, visited):
        if str(subject) in visited:
            return visited
        visited[str(subject)] = {}
        for predicate, obj in graph.predicate_objects(subject):
            if isinstance(obj, rdflib.Literal):
                visited[str(subject)][str(predicate)] = str(obj)
            elif isinstance(obj, rdflib.URIRef):
                if any(str(obj).startswith(prefix) for prefix in KNOWN_PREFIXES):
                    visited[str(subject)][str(predicate)] = str(obj)
                else:
                    traverse(obj, visited)
            elif isinstance(obj, rdflib.BNode):
                traverse(obj, visited)
        return visited
    return traverse(subject, {})

def create_text_from_literals(literals: Dict[str, Dict[str, str]]) -> str:
    return " ".join(
        f"{subject} - {predicates} -> {literal}"
        for subject, predicates in literals.items()
        for literal in predicates.values()
    )

def get_entity_texts(graph):
    entity_texts = {}
    for s in set(graph.subjects()):
        if isinstance(s, rdflib.term.BNode):
            continue
        dict_literals = traverse_graph_and_get_literals(graph, s)
        text = create_text_from_literals(dict_literals)
        if text:
            entity_texts[s] = text
    return entity_texts

# =============================
# ==== Code for the model =====
# =============================
def run_graph_embedding(triples_array):
    triples_factory = TriplesFactory.from_labeled_triples(triples_array)
    training_factory, testing_factory = triples_factory.split([0.8, 0.2])
    result = pipeline(
        training=training_factory,
        testing=testing_factory,
        model=GRAPH_MODEL,
        model_kwargs=dict(embedding_dim=TEXT_DIM),
        training_loop='slcwa',
        training_kwargs=dict(num_epochs=NUM_EPOCHS),
        evaluator_kwargs=dict(filtered=True),
    )
    return result.model, triples_factory.entity_to_id

# =============================
# ====== Hybrid vectors =======
# =============================
def get_hybrid_vector(entity, text_vec, graph_vec_dict):
    graph_vec = graph_vec_dict.get(str(entity), np.zeros(TEXT_DIM))
    text_vec = text_vec.cpu().numpy()
    return ALPHA * text_vec + (1 - ALPHA) * graph_vec

# =============================
# ========= Matching ==========
# =============================
def match_entities(ids1, ids2, hybrid_vecs1, hybrid_vecs2):
    sim_matrix = cosine_similarity(hybrid_vecs1, hybrid_vecs2)
    df = pd.DataFrame(sim_matrix, index=ids1, columns=ids2)
    matches = []
    for idx in df.index:
        max_sim = df.loc[idx].max()
        if max_sim >= THRESHOLD:
            best_match = df.loc[idx].idxmax()
            matches.append((idx, best_match, max_sim))
    return matches

# =============================
# ========== Main =============
# =============================
def main():
    combined_graph, phkg_graph, g2 = load_graph()
    entity_texts1 = get_entity_texts(phkg_graph)
    entity_texts2 = get_entity_texts(g2)
    ids1 = list(entity_texts1.keys())
    ids2 = list(entity_texts2.keys())

    model = SentenceTransformer(TEXT_MODEL)
    text_embeddings = {
        str(e): model.encode(text, convert_to_tensor=True)
        for e, text in {**entity_texts1, **entity_texts2}.items()
    }

    text_vectors1 = [text_embeddings[str(e)] for e in ids1]
    text_vectors2 = [text_embeddings[str(e)] for e in ids2]

    triples = extract_triples(combined_graph)
    triples_array = np.array(triples, dtype=str)

    model_graph, entity_to_id = run_graph_embedding(triples_array)
    graph_embedding_matrix = model_graph.entity_representations[0]().detach().cpu().numpy().real
    graph_embeddings = {e: graph_embedding_matrix[i] for e, i in entity_to_id.items()}

    hybrid_vecs1 = [get_hybrid_vector(e, t, graph_embeddings) for e, t in zip(ids1, text_vectors1)]
    hybrid_vecs2 = [get_hybrid_vector(e, t, graph_embeddings) for e, t in zip(ids2, text_vectors2)]

    matches = match_entities(ids1, ids2, hybrid_vecs1, hybrid_vecs2)
    match_json = [{"entity1": str(e1), "entity2": str(e2), "score": float(score)} for e1, e2, score in matches]

    with open("Distmatches.json", "w") as f:
        json.dump(match_json, f, indent=4)

if __name__ == "__main__":
    main()