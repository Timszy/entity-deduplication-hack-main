import json
import pandas as pd
import rdflib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import random
import os
# -------------------------------
# Helper functions for RDF extraction
# -------------------------------
# Known prefixes for linking URIs remain unchanged
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

# def traverse_graph_and_get_literals(graph, subject) -> Dict[str, Dict[str, str]]:
#     def traverse(subject, visited: Dict[str, Dict[str, str]]):
#         if str(subject) in visited:
#             return visited
#         visited[str(subject)] = {}
#         for predicate, obj in graph.predicate_objects(subject):
#             if isinstance(obj, rdflib.Literal):
#                 visited[str(subject)][str(predicate)] = str(obj)
#             elif isinstance(obj, rdflib.URIRef):
#                 if any(str(obj).startswith(prefix) for prefix in KNOWN_PREFIXES):
#                     visited[str(subject)][str(predicate)] = str(obj)
#                 else:
#                     traverse(obj, visited)
#             elif isinstance(obj, rdflib.BNode):
#                 traverse(obj, visited)
#             else:
#                 print(f"Unknown type: {type(obj)}")
#         return visited
#     return traverse(subject, {})

def traverse_graph_and_get_literals(
    graph, subject
) -> dict[str, dict[str, str]]:
    def traverse(
        subject: rdflib.URIRef | rdflib.BNode,
        visited: dict[str, dict[str, str]],
    ):
        if str(subject) in visited:
            return visited
        visited[str(subject)] = {}
        for predicate, obj in graph.predicate_objects(
            subject
        ):
            match type(obj):
                case rdflib.Literal:
                    visited[str(subject)][
                        str(predicate)
                    ] = str(obj)
                case rdflib.URIRef:
                    if any(
                        str(obj).startswith(prefix)
                        for prefix in KNOWN_PREFIXES
                    ):
                        visited[str(subject)][
                            str(predicate)
                        ] = str(obj)
                    else:
                        traverse(obj, visited
                                 )
                case rdflib.BNode:
                    traverse(obj, visited)
                case _:
                    print(f"Unknown type: {type(obj)}")
        return visited
    return traverse(subject, {})

def create_text_from_literals(literals: Dict[str, Dict[str, str]]) -> str:
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
    """
    literals = traverse_graph_and_get_literals(graph, subject)
    if str(subject) in literals:
        filtered = {pred: obj for pred, obj in literals[str(subject)].items()}
        return filtered
    return {}

# -------------------------------
# Multi-view Embedding Components
# -------------------------------
# 1. Name view: use SentenceTransformer on the full text (or just a label)
def get_name_view_embedding(entity, entity_texts, model):
    text = entity_texts.get(entity, "")
    if text:
        return model.encode(text)
    else:
        return np.zeros(model.get_sentence_embedding_dimension())

# 2. Relation view: already learned via TransE (graph_embeddings)
#    We assume graph_embeddings is a dict mapping entity IDs to their vector.


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# 3. Attribute view: using a simple CNN over attribute–value pairs.
class AttributeCNN(nn.Module):
    def __init__(self, d, num_filters=2, kernel_width=4, output_dim=384):
        super(AttributeCNN, self).__init__()
        # Input shape: (batch_size, 1, 2, d) where 2 corresponds to [attribute, value]
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(2, kernel_width))
        # Calculate the resulting width after convolution
        conv_out_width = d - kernel_width + 1
        self.fc = nn.Linear(num_filters * conv_out_width, output_dim)
        
    def forward(self, x):
        # x: (batch_size, 1, 2, d)
        x = self.conv(x)  # (batch_size, num_filters, 1, conv_out_width)
        x = F.relu(x)
        x = x.squeeze(2)  # (batch_size, num_filters, conv_out_width)
        x = x.view(x.size(0), -1)  # flatten: (batch_size, num_filters * conv_out_width)
        x = self.fc(x)
        x = torch.tanh(x)
        return x

def get_attribute_view_embedding(attributes, model, attribute_cnn, text_dim):
    """
    Process attribute–value pairs by encoding both the attribute (predicate) 
    and its value with the SentenceTransformer. Concatenate them into a 2 x d matrix,
    pass through the CNN, and average over all attributes.
    """
    pairs_embeddings = []
    for pred, value in attributes.items():
        # Encode attribute and value separately
        pred_emb = model.encode(pred)  # shape: (text_dim,)
        value_emb = model.encode(value)  # shape: (text_dim,)
        # Stack to get a matrix of shape (2, text_dim)
        pair_matrix = np.stack([pred_emb, value_emb], axis=0)
        pairs_embeddings.append(pair_matrix)
    if pairs_embeddings:
        all_attr_embeddings = []
        for pair_matrix in pairs_embeddings:
            # Convert to tensor and add a batch dimension: (1, 2, text_dim)
            tensor_pair = torch.tensor(pair_matrix, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                attr_emb = attribute_cnn(tensor_pair)  # (1, text_dim)
            all_attr_embeddings.append(attr_emb.squeeze(0).numpy())
        # Average over all attribute pairs
        return np.mean(all_attr_embeddings, axis=0)
    else:
        return np.zeros(text_dim)

def combine_embeddings(name_emb, rel_emb, attr_emb):
    """
    Combine view-specific embeddings using weighted view averaging as described in Equation (17)
    of the MultiKE paper. We compute cosine similarity of each view with the average embedding
    and use these as weights.
    """
    views = [name_emb, rel_emb, attr_emb]
    avg_emb = sum(views) / len(views)
    weights = []
    for v in views:
        sim = cosine_similarity(v.reshape(1, -1), avg_emb.reshape(1, -1))[0][0]
        weights.append(sim)
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)
    combined_emb = sum(w * v for w, v in zip(weights, views))
    return combined_emb

# -------------------------------
# Load RDF graphs and prepare data
# -------------------------------
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

# Replace with your file paths
g1.parse("data/healthcare_graph_original.ttl")
g2.parse("data/healthcare_graph_unstruct.ttl")
master_graph.parse("data/master_data.ttl")

phkg_graph = g1 + master_graph

alpha = 0.0  # (no longer used in multi-view)
text_dim = 384  # dimension for the SentenceTransformer
threshold = 0.4
num_epochs = 100  # For TransE training

# -------------------------------
# For graph embeddings (Relation view) using TransE via PyKEEN
# -------------------------------
def extract_triples(graph):
    """Extract triples from an RDF graph, skipping blank nodes."""
    triples = []
    for s, p, o in graph:
        if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
            continue
        triples.append((str(s), str(p), str(o)))
    return triples

# Combine the two graphs for training the graph embedding
combined_graph = phkg_graph + g2
triples = extract_triples(combined_graph)
triples_df = pd.DataFrame(triples, columns=["s", "p", "o"])
print(triples_df.head())
triples_array = triples_df[["s", "p", "o"]].values
triples_factory = TriplesFactory.from_labeled_triples(triples_array)
print(triples_factory)
train_tf, valid_tf, test_tf = triples_factory.split([0.8, 0.1, 0.1], random_state=42)
result = pipeline(
    training=train_tf,
    testing=test_tf,
    validation=valid_tf,
    model='ComplEx',
    model_kwargs=dict(embedding_dim=text_dim),
    training_kwargs=dict(num_epochs=num_epochs, use_tqdm_batch=False)
)
entity_to_id = train_tf.entity_to_id
embedding_matrix = result.model.entity_representations[0]._embeddings.weight.detach().cpu().numpy()
graph_embeddings = {entity: embedding_matrix[idx] for entity, idx in entity_to_id.items()}

# -------------------------------
# For text embeddings (Name view)
# -------------------------------
entity_texts1 = get_entity_texts(phkg_graph)
entity_texts2 = get_entity_texts(g2)
ids1 = list(entity_texts1.keys())
ids2 = list(entity_texts2.keys())

model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings1 = model.encode(list(entity_texts1.values()), convert_to_tensor=True)
text_embeddings2 = model.encode(list(entity_texts2.values()), convert_to_tensor=True)

# -------------------------------
# Initialize the AttributeCNN for the attribute view
# -------------------------------
# Here we set d = text_dim, kernel_width = 4, num_filters = 2, and output_dim = text_dim.
attribute_cnn = AttributeCNN(d=text_dim, num_filters=2, kernel_width=4, output_dim=text_dim)

# -------------------------------
# Create multi-view combined embeddings for each entity.
# -------------------------------
def compute_combined_embedding(entity, entity_texts, graph_source):
    # Name view: using the entity's text description
    name_emb = get_name_view_embedding(entity, entity_texts, model)
    # Relation view: from the TransE model (if not found, use zeros)
    rel_emb = graph_embeddings.get(str(entity), np.zeros(text_dim))
    # Attribute view: extract attributes from the graph and compute embedding using CNN
    attributes = get_attributes(graph_source, entity)
    attr_emb = get_attribute_view_embedding(attributes, model, attribute_cnn, text_dim)
    # Combine the three views
    combined_emb = combine_embeddings(name_emb, rel_emb, attr_emb)
    return combined_emb

combined_vectors1 = {}
for ent in ids1:
    combined_vectors1[str(ent)] = compute_combined_embedding(ent, entity_texts1, phkg_graph)

combined_vectors2 = {}
for ent in ids2:
    combined_vectors2[str(ent)] = compute_combined_embedding(ent, entity_texts2, g2)

# -------------------------------
# Matching based on cosine similarity between combined embeddings.
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
    """
    Match entities from two graphs based on similarity scores.
    """
    matches = []
    for idx in similarity_df.index:
        max_sim = similarity_df.loc[idx].max()
        if max_sim >= threshold:
            best_match = similarity_df.loc[idx].idxmax()
            matches.append((idx, best_match, max_sim))
    return matches

matched_entities = match_entities(df_similarity, threshold)

final_result = []

for ent1, ent2, score in matched_entities:
    # Get all literals for each entity from the corresponding graph.
    entity1_literals = traverse_graph_and_get_literals(phkg_graph, ent1)
    entity2_literals = traverse_graph_and_get_literals(g2, ent2)
    
    score_float = float(score)
    score_str = str(score_float)
    
    # For each entity, fetch the dictionary of predicate: object pairs.
    # Here we assume the main information for the entity is stored under its own string.
    # details1 = entity1_literals.get(str(ent1), {})
    # details2 = entity2_literals.get(str(ent2), {})
    

    if str(ent1) in entity1_literals:
        entity1_predicates = entity1_literals[str(ent1)]
    else:
        entity1_predicates = {}
        
    if str(ent2) in entity2_literals:
        entity2_predicates = entity2_literals[str(ent2)]
    else:
        entity2_predicates = {}

    all_predicates = sorted(set(list(entity1_predicates.keys()) + list(entity2_predicates.keys())))

    
    # Create detailed dictionaries that include both:
    # - A full copy of the literals dictionary (all predicates and objects)
    # - A list of sorted predicate/object pairs for easier inspection.
    # entity1_details = {
    #     "from": "phkg_graph",
    #     "subject": str(ent1),
    #     "literals": details1,
    #     "predicates": [{"predicate": pred, "object": details1[pred]} 
    #                    for pred in sorted(details1.keys())]
    # }
    
    # entity2_details = {
    #     "from": "g2",
    #     "subject": str(ent2),
    #     "literals": details2,
    #     "predicates": [{"predicate": pred, "object": details2[pred]} 
    #                    for pred in sorted(details2.keys())]
    # }
    
    # duplication_type = "exact" if score_float >= 0.9 else "similar" if score_float >= 0.7 else "conflict"
    # final_result.append({
    #     "entities": [{"entity1": entity1_details}, {"entity2": entity2_details}],
    #     "similarity_score": score_str,
    #     "duplication_type": duplication_type,
    # })

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


os.makedirs("matches", exist_ok=True)
output_file = os.path.join("matches", "matchestest.json")
with open(output_file, "w") as f:
    json.dump(final_result, f, indent=4)
