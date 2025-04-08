import json
import pandas as pd
import rdflib
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Global settings and constants
# -------------------------------
text_dim = 384      # Dimension for SentenceTransformer embeddings
threshold = 0.7     # Matching threshold for deduplication
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

# -------------------------------
# RDF and Graph Helper Functions
# -------------------------------

def rdf_to_nx(graph):
    """Convert an RDF graph to a networkx graph based on subject-object edges."""
    G = nx.Graph()
    for s, p, o in graph:
        if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
            continue
        G.add_edge(str(s), str(o), predicate=str(p))
    return G

def get_graph_embeddings(graph, dimensions=text_dim):
    """
    Use node2vec to create a relation view embedding (structure-based).
    """
    G_nx = rdf_to_nx(graph)
    node2vec = Node2Vec(G_nx, dimensions=dimensions, walk_length=10, num_walks=80, workers=1)
    model = node2vec.fit()
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    return embeddings

# -------------------------------
# Text Extraction Functions
# -------------------------------

def traverse_graph_and_get_literals(graph, subject) -> Dict[str, Dict[str, str]]:
    """
    Traverse the RDF graph starting at subject and collect predicate-object pairs.
    """
    def traverse(subject, visited: Dict[str, Dict[str, str]]):
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
            else:
                print(f"Unknown type: {type(obj)}")
        return visited
    return traverse(subject, {})

def create_text_from_literals(literals: Dict[str, Dict[str, str]]) -> str:
    """
    Combine the literals into a single string.
    """
    return " ".join([
        f"{subject} - {predicates} -> {literal}"
        for subject, predicates in literals.items()
        for literal in predicates.values()
    ])

def get_entity_texts(graph):
    """
    Extract entities and their textual descriptions from an RDF graph.
    """
    entity_texts = {}
    for s in set(graph.subjects()):
        if isinstance(s, rdflib.term.BNode):
            continue
        dict_literals = traverse_graph_and_get_literals(graph, s)
        text = create_text_from_literals(dict_literals)
        if text:
            entity_texts[s] = text
        print(f"Entity {s} with text: {text}")
    return entity_texts

def get_attributes(graph, subject) -> Dict[str, str]:
    """
    Extract attribute-related literal values for a subject.
    Here we use the literals directly attached to the subject.
    Optionally, you can filter out predicates used solely for names.
    """
    literals = traverse_graph_and_get_literals(graph, subject)
    return literals.get(str(subject), {})  # Return dictionary of predicate: object for the subject

# -------------------------------
# Multi-View Embedding Components
# -------------------------------

# 1. Name View Embedding: use SentenceTransformer to encode the textual description.
def get_name_view_embedding(entity, entity_texts, model):
    text = entity_texts.get(entity, "")
    if text:
        return model.encode(text)
    else:
        return np.zeros(model.get_sentence_embedding_dimension())

# 2. Relation View Embedding: from node2vec (already computed as graph_embeddings).
# No additional function is needed; we look up graph_embeddings by entity ID.

# 3. Attribute View Embedding: use a CNN to process attribute (predicate, object) pairs.
class AttributeCNN(nn.Module):
    def __init__(self, d, num_filters=2, kernel_width=4, output_dim=text_dim):
        """
        d: input dimension (same as text_dim)
        """
        super(AttributeCNN, self).__init__()
        # Input shape for each attribute pair will be (batch_size, 1, 2, d)
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(2, kernel_width))
        conv_out_width = d - kernel_width + 1
        self.fc = nn.Linear(num_filters * conv_out_width, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 2, d)
        x = self.conv(x)  # shape: (batch_size, num_filters, 1, conv_out_width)
        x = F.relu(x)
        x = x.squeeze(2)  # shape: (batch_size, num_filters, conv_out_width)
        x = x.view(x.size(0), -1)  # flatten: (batch_size, num_filters * conv_out_width)
        x = self.fc(x)
        x = torch.tanh(x)
        return x

def get_attribute_view_embedding(attributes, model, attribute_cnn, text_dim):
    """
    Encode attribute-value pairs.
    For each attribute pair, encode the predicate and the object using the SentenceTransformer,
    stack the embeddings to create a 2 x d matrix, add channel dimensions, and pass through the CNN.
    Average the output over all attribute pairs.
    """
    pairs_embeddings = []
    for pred, value in attributes.items():
        pred_emb = model.encode(pred)       # shape: (text_dim,)
        value_emb = model.encode(value)       # shape: (text_dim,)
        pair_matrix = np.stack([pred_emb, value_emb], axis=0)  # shape: (2, text_dim)
        pairs_embeddings.append(pair_matrix)
    if pairs_embeddings:
        all_attr_embeddings = []
        for pair_matrix in pairs_embeddings:
            # Create tensor with shape (1, 1, 2, text_dim)
            tensor_pair = torch.tensor(pair_matrix, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                attr_emb = attribute_cnn(tensor_pair)  # Expected output shape: (1, text_dim)
            all_attr_embeddings.append(attr_emb.squeeze(0).numpy())
        return np.mean(all_attr_embeddings, axis=0)
    else:
        return np.zeros(text_dim)

def combine_embeddings(name_emb, rel_emb, attr_emb):
    """
    Combine view-specific embeddings via weighted averaging.
    We compute the cosine similarity between each view and the average embedding,
    then weight each view by these similarities.
    """
    views = [name_emb, rel_emb, attr_emb]
    avg_emb = sum(views) / len(views)
    weights = []
    for v in views:
        sim = cosine_similarity(v.reshape(1, -1), avg_emb.reshape(1, -1))[0][0]
        weights.append(sim)
    weights = np.array(weights)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)
    combined_emb = sum(w * v for w, v in zip(weights, views))
    return combined_emb

# -------------------------------
# Load RDF Graphs and Prepare Data
# -------------------------------

# Load your RDF graphs
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

g1.parse("data/healthcare_graph_original.ttl")
g2.parse("data/healthcare_graph_unstruct.ttl")
master_graph.parse("data/master_data.ttl")

# Combine graphs for the source; we use g1 + master_graph for one set
phkg_graph = g1 + master_graph

# For the relation view, we use the combined graph of phkg_graph and g2
combined_graph = phkg_graph + g2

# -------------------------------
# Compute Graph Embeddings using node2vec (Relation View)
# -------------------------------
graph_embeddings = get_graph_embeddings(combined_graph, dimensions=text_dim)

# -------------------------------
# Compute Text (Name View) Embeddings
# -------------------------------
entity_texts1 = get_entity_texts(phkg_graph)
entity_texts2 = get_entity_texts(g2)
ids1 = list(entity_texts1.keys())
ids2 = list(entity_texts2.keys())

model = SentenceTransformer("all-MiniLM-L6-v2")
# (You may use model.encode on texts later when needed for each entity.)

# -------------------------------
# Initialize the AttributeCNN for the Attribute View
# -------------------------------
attribute_cnn = AttributeCNN(d=text_dim, num_filters=2, kernel_width=4, output_dim=text_dim)

# -------------------------------
# Compute Multi-View Combined Embeddings for Each Entity
# -------------------------------
def compute_multi_view_embedding(entity, entity_texts, graph_source):
    # Name view: textual embedding from the entity text description.
    name_emb = get_name_view_embedding(entity, entity_texts, model)
    # Relation view: from the graph embedding (node2vec)
    rel_emb = graph_embeddings.get(str(entity), np.zeros(text_dim))
    # Attribute view: extract attribute information from the graph.
    attributes = get_attributes(graph_source, entity)
    attr_emb = get_attribute_view_embedding(attributes, model, attribute_cnn, text_dim)
    # Combine the three views
    combined_emb = combine_embeddings(name_emb, rel_emb, attr_emb)
    return combined_emb

combined_vectors1 = {}
for ent in ids1:
    combined_vectors1[str(ent)] = compute_multi_view_embedding(ent, entity_texts1, phkg_graph)

combined_vectors2 = {}
for ent in ids2:
    combined_vectors2[str(ent)] = compute_multi_view_embedding(ent, entity_texts2, g2)

# -------------------------------
# Matching: Compare Combined Embeddings via Cosine Similarity
# -------------------------------
ents1 = list(combined_vectors1.keys())
ents2 = list(combined_vectors2.keys())
matrix1 = np.array([combined_vectors1[e] for e in ents1])
matrix2 = np.array([combined_vectors2[e] for e in ents2])
similarity_matrix = cosine_similarity(matrix1, matrix2)

df_similarity = pd.DataFrame(similarity_matrix, index=entity_texts1, columns=entity_texts2)
print("Combined similarity matrix shape:", df_similarity.shape)
print("Combined similarity sample (first 5 rows):")
print(df_similarity.head())

def match_entities(similarity_df, threshold):
    """Match entities based on similarity scores."""
    matches = []
    for idx in similarity_df.index:
        max_sim = similarity_df.loc[idx].max()
        if max_sim >= threshold:
            best_match = similarity_df.loc[idx].idxmax()
            matches.append((idx, best_match, max_sim))
    return matches

matched_entities = match_entities(df_similarity, threshold)

# -------------------------------
# Output Results with Detailed Predicates and Objects
# -------------------------------
final_result = []

for ent1, ent2, score in matched_entities:
    entity1_literals = traverse_graph_and_get_literals(phkg_graph, ent1)
    entity2_literals = traverse_graph_and_get_literals(g2, ent2)
    
    score_float = float(score)
    score_str = str(score_float)
    
    if str(ent1) in entity1_literals:
        entity1_predicates = entity1_literals[str(ent1)]
    else:
        entity1_predicates = {}
        
    if str(ent2) in entity2_literals:
        entity2_predicates = entity2_literals[str(ent2)]
    else:
        entity2_predicates = {}

    all_predicates = sorted(set(list(entity1_predicates.keys()) + list(entity2_predicates.keys())))
    
    entity1_details = {
        "from": "phkg_graph",
        "subject": str(ent1),
        #"literals": entity1_predicates,
        "predicates": [{"predicate": pred, "object": entity1_predicates.get(pred, "N/A")}
                       for pred in all_predicates if pred in entity1_predicates]
    }
    entity2_details = {
        "from": "g2",
        "subject": str(ent2),
        #"literals": entity2_predicates,
        "predicates": [{"predicate": pred, "object": entity2_predicates.get(pred, "N/A")}
                       for pred in all_predicates if pred in entity2_predicates]
    }
    
    duplication_type = "exact" if score_float >= 0.9 else "similar" if score_float >= 0.7 else "conflict"
    
    final_result.append({
        "entities": [{"entity1": entity1_details}, {"entity2": entity2_details}],
        "similarity_score": score_str,
        "duplication_type": duplication_type,
    })

with open("matches3Un.json", "w") as f:
    json.dump(final_result, f, indent=4)
