import json
import pandas as pd
import rdflib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from typing import Dict
from rdflib.namespace import RDF, XSD
from urllib.parse import urlparse
import torch.nn.functional as F

# ---------------------------
# Load RDF Graphs
# ---------------------------
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

g1.parse("data/healthcare_graph_original_v2.ttl")
g2.parse("data/prog_data/healthcare_graph_var_only_fixed.ttl")
master_graph.parse("data/master_data.ttl")

phkg_graph = g1 + master_graph

# ---------------------------
# Configuration
# ---------------------------
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
WEAK_PREDICATES = {"schema:identifier"}
ALPHA = 0.5

# ---------------------------
# Helper Functions
# ---------------------------
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
            texts[s] = (text, type_label)
    return texts

def group_by_type(entity_dict):
    grouped = {}
    for entity, (text, typ) in entity_dict.items():
        grouped.setdefault(typ, []).append((entity, text))
    return grouped

def extract_triples(graph):
    return [
        (str(s), str(p), str(o))
        for s, p, o in graph
        if not isinstance(s, rdflib.BNode) and not isinstance(o, rdflib.BNode)
    ]

def extract_literals_for_literalE(graph):
    literals = {}
    for s in graph.subjects():
        if isinstance(s, rdflib.BNode):
            continue
        literal_feats = {}
        for p, o in graph.predicate_objects(s):
            if isinstance(o, rdflib.Literal) and o.datatype in [None, XSD.string]:
                pred = get_prefixed_predicate(str(p))
                literal_feats[pred] = str(o)
        if literal_feats:
            literals[str(s)] = literal_feats
    return literals

def get_hybrid_vector(entity_uri, text_vec, graph_vec_dict):
    graph_vec = graph_vec_dict.get(str(entity_uri), np.zeros_like(text_vec.cpu().numpy()))
    text_np = text_vec.cpu().numpy()
    return ALPHA * text_np + (1 - ALPHA) * graph_vec

def match_entities(similarity_df, threshold=0.7, top_k=2):
    matches = []
    for g2_entity in similarity_df.columns:
        top_matches = similarity_df[g2_entity].nlargest(top_k)
        for phkg_entity, sim in top_matches.items():
            if sim >= threshold:
                matches.append((phkg_entity, g2_entity, sim))
    return matches

# ---------------------------
# Main Execution Pipeline
# ---------------------------
print("Preparing text representations...")
entity_texts1 = get_entity_texts(phkg_graph)
entity_texts2 = get_entity_texts(g2)

grouped1 = group_by_type(entity_texts1)
grouped2 = group_by_type(entity_texts2)

print("Training LiteralE model...")
triples = extract_triples(phkg_graph + g2)
triples_array = np.array(triples, dtype=str)
literal_features = extract_literals_for_literalE(phkg_graph + g2)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

result = pipeline(
    model='LiteralE',
    model_kwargs={'base_model': 'DistMult', 
                  'literals': literal_features},
    training=triples_factory,
    training_kwargs={'num_epochs': 50},
    stopper='early',
    stopper_kwargs={'frequency': 5, 'patience': 3},
)

model_graph = result.model
entity_to_id = triples_factory.entity_to_id
embedding_matrix = model_graph.entity_representations[0]().detach().cpu().numpy()
graph_embeddings = {e: embedding_matrix[i] for e, i in entity_to_id.items()}

print("Generating hybrid embeddings and similarity matrix...")
all_matches = []
for typ in set(grouped1) & set(grouped2):
    ids1, texts1 = zip(*grouped1[typ])
    ids2, texts2 = zip(*grouped2[typ])

    emb1 = model.encode(texts1, convert_to_tensor=True)
    emb2 = model.encode(texts2, convert_to_tensor=True)

    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    hybrid_vecs1 = [get_hybrid_vector(e, v, graph_embeddings) for e, v in zip(ids1, emb1)]
    hybrid_vecs2 = [get_hybrid_vector(e, v, graph_embeddings) for e, v in zip(ids2, emb2)]

    sim_matrix = cosine_similarity(hybrid_vecs1, hybrid_vecs2)
    df_sim = pd.DataFrame(sim_matrix, index=ids1, columns=ids2)
    all_matches.append(df_sim)

df_similarity_all = pd.concat(all_matches)
print("Similarity matrix shape:", df_similarity_all.shape)

matched_entities = match_entities(df_similarity_all)
print("Matched entities:", len(matched_entities))

final_result = []
for ent1, ent2, score in matched_entities:
    entity1_literals = traverse_graph_and_get_literals(phkg_graph, ent1)
    entity2_literals = traverse_graph_and_get_literals(g2, ent2)
    score = float(score)
    score_str = str(score)

    entity1_predicates = entity1_literals.get(str(ent1), {})
    entity2_predicates = entity2_literals.get(str(ent2), {})
    all_predicates = sorted(set(entity1_predicates.keys()) | set(entity2_predicates.keys()))

    entity1_details = {
        "from": "phkg_graph",
        "subject": str(ent1),
        "predicates": [{"predicate": pred, "object": entity1_predicates.get(pred, "N/A")} for pred in all_predicates]
    }
    entity2_details = {
        "from": "g2",
        "subject": str(ent2),
        "predicates": [{"predicate": pred, "object": entity2_predicates.get(pred, "N/A")} for pred in all_predicates]
    }

    duplication_type = "exact" if score >= 0.9 else "similar" if score >= 0.7 else "conflict"
    final_result.append({
        "entities": [
            {"entity1": entity1_details},
            {"entity2": entity2_details}
        ],
        "similarity_score": score_str,
        "duplication_type": duplication_type
    })

with open("revisedHackyg2.json", "w") as f:
    json.dump(final_result, f, indent=4)
print("Final result saved to revisedHackyg2.json")

