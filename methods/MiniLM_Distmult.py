import json
import pandas as pd
import rdflib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from rdflib.namespace import RDF
from urllib.parse import urlparse
import torch.nn.functional as F

# =============================
# ========= Configs ===========
# =============================
TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_DIM = 384
THRESHOLD = 0.7
NUM_EPOCHS = 100
GRAPH_MODEL = "DistMult"
ALPHA = 0.5 # You can change this value to weight the text embedding (0.0 = is graph only)
WEAK_PREDICATES = {"schema:identifier"}

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
def get_prefixed_predicate(uri):
    if uri.startswith("http://schema.org/") or uri.startswith("https://schema.org/"):
        return "schema:" + uri.split("/")[-1]
    else:
        return urlparse(uri).fragment or uri.split("/")[-1]

def traverse_graph_and_get_literals(graph, subject) -> dict[str, dict[str, str]]:
    def traverse(subject, visited):
        if str(subject) in visited:
            return visited
        visited[str(subject)] = {}
        for predicate, obj in graph.predicate_objects(subject):
            pred_str = get_prefixed_predicate(str(predicate))
            if isinstance(obj, rdflib.Literal) and pred_str not in WEAK_PREDICATES:
                visited[str(subject)][pred_str] = str(obj)
            elif isinstance(obj, (rdflib.URIRef, rdflib.BNode)):
                traverse(obj, visited)
        return visited
    return traverse(subject, {})

def create_text_from_literals(subject_uri: str, literals: dict[str, dict[str, str]], graph: rdflib.Graph) -> str:
    parts = []
    for o in graph.objects(rdflib.URIRef(subject_uri), RDF.type):
        parts.append(f"[{get_prefixed_predicate(str(o))}]")
        break
    for _, preds in literals.items():
        for pred, val in preds.items():
            parts.append(f"{pred}: {val}")
    return " ".join(parts)

def get_entity_texts(graph):
    texts = {}
    for s in set(graph.subjects()):
        if isinstance(s, rdflib.BNode):
            continue
        literals = traverse_graph_and_get_literals(graph, s)
        text = create_text_from_literals(str(s), literals, graph)
        type_label = None
        for o in graph.objects(rdflib.URIRef(s), RDF.type):
            type_label = get_prefixed_predicate(str(o))
            break
        if text and type_label:
            texts[s] = text
    return texts

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
        batch_size=32,
        random_seed=69
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
def match_entities(ids1, ids2, hybrid_vecs1, hybrid_vecs2, top_k=2):
    """
    Match entities from two sets of embedding‐vectors by taking the top_k 
    candidates in each column of the similarity‐matrix, then filtering by threshold.

    Args:
        ids1 (List[str]):   List of entity IDs corresponding to hybrid_vecs1 (rows).
        ids2 (List[str]):   List of entity IDs corresponding to hybrid_vecs2 (columns).
        hybrid_vecs1 (ndarray):  Shape = (len(ids1), D)
        hybrid_vecs2 (ndarray):  Shape = (len(ids2), D)
        threshold (float):  Minimum cosine‐similarity to count as a “match.”
        top_k (int):        How many top candidates per column to consider.

    Returns:
        List[Tuple[str, str, float]]:
            Each tuple is (id_from_ids1, id_from_ids2, similarity_score),
            where similarity_score ≥ threshold.  At most top_k matches are 
            returned for each id in ids2.
    """
    sim_matrix = cosine_similarity(hybrid_vecs1, hybrid_vecs2)
    df = pd.DataFrame(sim_matrix, index=ids1, columns=ids2)
    matches = []
    for g2_entity in df.columns:
        #    a) Find the top_k rows (ids1) by similarity in this column
        top_matches = df[g2_entity].nlargest(top_k)

        #    b) Filter out any whose similarity < threshold
        for phkg_entity, sim in top_matches.items():
            if sim >= THRESHOLD:
                matches.append((phkg_entity, g2_entity, sim))

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

    matched_entities = match_entities(ids1, ids2, hybrid_vecs1, hybrid_vecs2)
    
    final_result = []

    for ent1, ent2, score in matched_entities:
        entity1_literals = traverse_graph_and_get_literals(phkg_graph, ent1)
        entity2_literals = traverse_graph_and_get_literals(g2, ent2)

        # Convert to normal Python float
        score = float(score)
        score_str = str(score)

        # Get the predicates from the subject
        if str(ent1) in entity1_literals:
            entity1_predicates = entity1_literals[str(ent1)]
        else:
            entity1_predicates = {}
        
        if str(ent2) in entity2_literals:
            entity2_predicates = entity2_literals[str(ent2)]
        else:
            entity2_predicates = {}
    
        # Get all predicates from both entities to ensure consistent order
        all_predicates = sorted(set(list(entity1_predicates.keys()) + list(entity2_predicates.keys())))
    
        # Create entity details with sorted predicates
        entity1_details = {
        "from": "phkg_graph",
        "subject": str(ent1),
        "predicates": [
            {
                "predicate": pred,
                "object": entity1_predicates.get(pred, "N/A")
            }
            for pred in all_predicates
            if pred in entity1_predicates
        ]
        }

        entity2_details = {
        "from": "g2",
        "subject": str(ent2),
        "predicates": [
            {
                "predicate": pred,
                "object": entity2_predicates.get(pred, "N/A")
            }
            for pred in all_predicates
            if pred in entity2_predicates
        ]
        }

        duplication_type = (
            "exact" if score >= 0.9 else "similar" if score >= 0.7 else "conflict"
        )

        final_result.append(
        {
            "entities": [
                {"entity1": entity1_details},
                {"entity2": entity2_details}
            ],
            "similarity_score": score_str,
            "duplication_type": duplication_type,
        }
        )

        with open("matches/sen.json", "w") as f:
            json.dump(final_result, f, indent=4)

if __name__ == "__main__":
    main()