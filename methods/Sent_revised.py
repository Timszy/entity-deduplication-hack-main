import json
import pandas as pd
import rdflib
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
from typing import Dict
from rdflib.namespace import RDF
from urllib.parse import urlparse
import time # Import time module
import torch.nn.functional as F
#                     for prefix in KNOWN_PREFIXES

# Load the RDF graphs
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

# Replace 'graph1.rdf' and 'graph2.rdf' with the paths to your RDF files
g1.parse("data/healthcare_graph_original_v2.ttl")
g2.parse("data/prog_data/healthcare_graph_var_only_fixed.ttl")
master_graph.parse("data/master_data.ttl")

phkg_graph = g1 + master_graph
# ---------------------------
# Configuration
# ---------------------------
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
WEAK_PREDICATES = {"schema:identifier"}

# ---------------------------
# Helpers
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

# ---------------------------
# Apply to Your Graphs
# ---------------------------
entity_texts1 = get_entity_texts(phkg_graph)
entity_texts2 = get_entity_texts(g2)

grouped1 = group_by_type(entity_texts1)
grouped2 = group_by_type(entity_texts2)

all_matches = []
for typ in set(grouped1) & set(grouped2):
    ids1, texts1 = zip(*grouped1[typ])
    ids2, texts2 = zip(*grouped2[typ])

    emb1 = model.encode(texts1, convert_to_tensor=True)
    emb2 = model.encode(texts2, convert_to_tensor=True)

    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    sim_matrix = cosine_similarity(emb1.cpu(), emb2.cpu())
    df_sim = pd.DataFrame(sim_matrix, index=ids1, columns=ids2)

    all_matches.append(df_sim)

df_similarity_all = pd.concat(all_matches)

# Function to match entities based on similarity threshold
def match_entities(similarity_df, threshold=0.7):
    """
    Match entities from two graphs based on similarity scores.

    Args:
        similarity_df (pd.DataFrame): DataFrame of similarity scores.
        threshold (float): Similarity threshold for matching.

    Returns:
        list: A list of matched entity pairs and their similarity scores.
    """
    matches = []
    for idx in similarity_df.index:
        # Get the most similar entity and its score
        max_sim = similarity_df.loc[idx].max()
        if max_sim >= threshold:
            best_match = similarity_df.loc[idx].idxmax()
            matches.append((idx, best_match, max_sim))
    return matches


# Perform the matching
matched_entities = match_entities(
    df_similarity_all, threshold=0.7
)

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
    
with open("revisedHacky.json", "w") as f:
    json.dump(final_result, f, indent=4)