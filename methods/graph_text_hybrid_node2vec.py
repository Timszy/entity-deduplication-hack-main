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
import torch
import torch.nn.functional as F
import difflib
import re


# Load the RDF graphs
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

# Replace 'graph1.rdf' and 'graph2.rdf' with the paths to your RDF files
g1.parse("data/healthcare_graph_original_v2.ttl")
g2.parse("data/prog_data/healthcare_graph_var_only.ttl")
master_graph.parse("data/master_data.ttl")

phkg_graph = g1 + master_graph

# Start timer for the algorithm
algorithm_start_time = time.time()

alpha = 0.5 # You can change this value to weight the text embedding (0.0 = is graph only)
text_dim = 384 
threshold = 0.6 # Similarity threshold for matching
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
WEAK_PREDICATES = {"schema:identifier"}

# ---------------------------
# Graph functions
# ---------------------------

# For graph embeddings
# Convvert rdf to nx for the node2vec
def rdf_to_nx(graph):
    G = nx.Graph()
    for s, p, o in graph:
        if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
            continue
        G.add_edge(str(s), str(o), predicate=str(p))
    return G

# Graph embedding
def get_graph_embeddings(graph, dimensions=text_dim):
    G_nx = rdf_to_nx(graph)
    node2vec = Node2Vec(G_nx, dimensions=dimensions, walk_length=10, num_walks=80, workers=1)
    model = node2vec.fit()
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    return embeddings
# ---------------------------
# Text Processing Functions
# ---------------------------

def camel_to_title(s: str) -> str:
    """
    Turn a camelCase (or mixedCase) string into Title Case.
    Ex:
      "jobTitle"  → "Job Title"
      "birthDate" → "Birth Date"
    """
    # Insert a space between a lowercase letter and uppercase letter
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return spaced.title()

def get_human_label(curie: str) -> str:
    """
    Given something like "schema:jobTitle" or "ex:somePredicate",
    split off the prefix (before ':') and turn the local-part into Title Case.
    If there's no ":", just run the entire string through camel_to_title.
    """
    if ":" in curie:
        _, local = curie.split(":", 1)
    else:
        local = curie
    if local.startswith("[") and local.endswith("]"):
        # If it's a list-like string, remove the brackets
        local = local[1:-1]
    return camel_to_title(local)

def get_prefixed_predicate(uri: str) -> str:
    """
    If it's schema.org, return "schema:<local>",
    else return the fragment or last path segment.
    """
    if uri.startswith(("http://schema.org/", "https://schema.org/")):
        return "schema:" + uri.split("/")[-1]
    else:
        frag = urlparse(uri).fragment
        return frag if frag else uri.split("/")[-1]

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

def create_text_from_literals(subject_uri: str,
                              literals: dict[str, dict[str, str]],
                              graph: rdflib.Graph) -> str:
    parts = []
    # 1) Show the rdf:type as "Type: <Human Label>."
    for o in graph.objects(rdflib.URIRef(subject_uri), RDF.type):
        curie = get_prefixed_predicate(str(o))
        human_type = get_human_label(curie)
        parts.append(f"Type: {human_type}.")
        break

    # 2) For each collected literal, replace "pred" with get_human_label(pred).
    #    Previously you did: parts.append(f"{pred}: {val}")
    #    Now do:          parts.append(f"{get_human_label(pred)}: {val}.")
    for _, preds in literals.items():
        for pred, val in preds.items():
            human_pred = get_human_label(pred)
            parts.append(f"{human_pred}: {val}.")

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

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Extract entities and group by type
entity_texts1 = get_entity_texts(phkg_graph)
entity_texts2 = get_entity_texts(g2)
grouped1 = group_by_type(entity_texts1)
grouped2 = group_by_type(entity_texts2)

# Combine both graphs for graph embeddings
print("Computing graph embeddings...")
combined_graph = phkg_graph + g2
graph_embeddings = get_graph_embeddings(combined_graph, dimensions=text_dim)

def get_hybrid_vector(entity, text_vector):
    graph_vector = graph_embeddings.get(str(entity), np.zeros(text_dim))
    text_vector_np = text_vector.cpu().numpy()
    return alpha * text_vector_np + (1 - alpha) * graph_vector

all_matches = []
for typ in set(grouped1) & set(grouped2):
    ids1, texts1 = zip(*grouped1[typ])
    ids2, texts2 = zip(*grouped2[typ])
    
    # Compute sentence embeddings
    emb1 = model.encode(texts1, convert_to_tensor=True)
    emb2 = model.encode(texts2, convert_to_tensor=True)
    
    # Get hybrid vectors for this type
    hybrid_vecs1 = [get_hybrid_vector(e, t) for e, t in zip(ids1, emb1)]
    hybrid_vecs2 = [get_hybrid_vector(e, t) for e, t in zip(ids2, emb2)]
    
    # Normalize
    hybrid_vecs1 = F.normalize(torch.tensor(hybrid_vecs1), p=2, dim=1)
    hybrid_vecs2 = F.normalize(torch.tensor(hybrid_vecs2), p=2, dim=1)
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(hybrid_vecs1.cpu(), hybrid_vecs2.cpu())
    df_sim = pd.DataFrame(sim_matrix, index=ids1, columns=ids2)
    
    all_matches.append(df_sim)

df_similarity_all = pd.concat(all_matches)

# Function to match entities based on similarity threshold
def match_entities(similarity_df, threshold=0.7, top_k=2):
    """
    Match entities from two graphs based on similarity scores.

    Args:
        similarity_df (pd.DataFrame): DataFrame of similarity scores.
        threshold (float): Similarity threshold for matching.

    Returns:
        list: A list of matched entity pairs and their similarity scores.
    """
    matches = []
    # Loop over g2 entities (columns), compare to phkg (rows)
    for g2_entity in similarity_df.columns:
        top_matches = similarity_df[g2_entity].nlargest(top_k)
        for phkg_entity, sim in top_matches.items():
            if sim >= threshold:
                matches.append((phkg_entity, g2_entity, sim))
    return matches

print("Matching entities based on similarity scores...")
# Perform the matching
matched_entities = match_entities(
    df_similarity_all
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

# Post processing: filter out duplicates based on literal similarity
def normalized_levenshtein(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()
print("Filtering matches based on predicate similarity...")
LEVENSHTEIN_THRESHOLD = 0.63  # Adjust as needed

filtered_result = []
for match in final_result:
    entity1 = match["entities"][0]["entity1"]
    entity2 = match["entities"][1]["entity2"]

    # Only compare predicates that both entities have
    preds1 = {p["predicate"]: p["object"] for p in entity1["predicates"] if p["object"] != "N/A"}
    preds2 = {p["predicate"]: p["object"] for p in entity2["predicates"] if p["object"] != "N/A"}
    common_preds = set(preds1.keys()) & set(preds2.keys())

    if not common_preds:
        continue

    sim_scores = []
    for pred in common_preds:
        sim = normalized_levenshtein(str(preds1[pred]).lower(), str(preds2[pred]).lower())
        sim_scores.append(sim)

    avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0

    if avg_sim >= LEVENSHTEIN_THRESHOLD:
        filtered_result.append(match)

print(f"Filtered matches: {len(filtered_result)} (from {len(final_result)})")

with open("matches/Node2vec_filtered.json", "w") as f:
    json.dump(filtered_result, f, indent=4)
print("Filtered result saved to SentRevisedv3_filtered.json")

