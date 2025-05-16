import json
import pandas as pd
import rdflib
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
from typing import Dict
import time # Import time module
from evaluate_helper import print_detailed_statistics, calculate_entity_level_metrics


# Load the RDF graphs
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

# Replace 'graph1.rdf' and 'graph2.rdf' with the paths to your RDF files
g1.parse("example_data/healthcare_graph_original_v2.ttl")
g2.parse("example_data/healthcare_graph_replaced_v2.ttl")
master_graph.parse("data/master_data.ttl")

phkg_graph = g1 + master_graph

# Start timer for the algorithm
algorithm_start_time = time.time()

alpha = 0.5 # You can change this value to weight the text embedding (0.0 = is graph only)
text_dim = 384 # Dim for the all-MiniLM-L6-v2
threshold = 0.50 # Similarity threshold for matching

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

# For text embeddings
def traverse_graph_and_get_literals(
    graph, subject
) -> Dict[str, Dict[str, str]]:
    def traverse(
        subject: rdflib.URIRef or rdflib.BNode, # type: ignore
        visited: Dict[str, Dict[str, str]],
    ):
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


def create_text_from_literals(
    literals: Dict[str, Dict[str, str]],
) -> str:
    return " ".join(
        [
            f"{subject} - {predicates} -> {literal}"
            for subject, predicates in literals.items()
            for literal in predicates.values()
        ]
    )


def get_entity_texts(graph):
    """
    Extract entities and their textual descriptions from an RDF graph.

    Args:
        graph (rdflib.Graph): The RDF graph.

    Returns:
        dict: A dictionary mapping entities to their text descriptions.
    """
    entity_texts = {}
    for s in set(graph.subjects()):
        # Skip blank nodes
        if isinstance(s, rdflib.term.BNode):
            continue

        # Get labels and comments (you can include other properties if needed)
        dict_literals = traverse_graph_and_get_literals(
            graph, s
        )
        text = create_text_from_literals(dict_literals)

        if text:
            entity_texts[s] = text

        print(f"Entity {s} with text: {text}")

    return entity_texts


# Extract entities and texts from both graphs
entity_texts1 = get_entity_texts(phkg_graph)
entity_texts2 = get_entity_texts(g2)

# Prepare the lists of texts for embedding
texts1 = list(entity_texts1.values())
texts2 = list(entity_texts2.values())
ids1 = list(entity_texts1.keys())
ids2 = list(entity_texts2.keys())

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
text_embeddings1 = model.encode(texts1, convert_to_tensor=True)
text_embeddings2 = model.encode(texts2, convert_to_tensor=True)


# For graph embeddings
# Combine the two graphs 
combined_graph = phkg_graph + g2
graph_embeddings = get_graph_embeddings(combined_graph, dimensions=text_dim)

# To get the ratio between the 2 embeddings 
def get_hybrid_vector(entity, text_vector):
    graph_vector = graph_embeddings.get(str(entity), np.zeros(text_dim))
    text_vector_np = text_vector.cpu().numpy()  # Convert tensor to NumPy array on CPU
    return alpha * text_vector_np + (1 - alpha) * graph_vector

# Combine the two embeddings
hybrid_vectors1 = [get_hybrid_vector(e, t) for e, t in zip(ids1, text_embeddings1)]
hybrid_vectors2 = [get_hybrid_vector(e, t) for e, t in zip(ids2, text_embeddings2)]

# Cosine similarity 
similarity_matrix = cosine_similarity(hybrid_vectors1, hybrid_vectors2)

# Convert similarity matrix to DataFrame for easier handling
df_similarity = pd.DataFrame(
    similarity_matrix,
    index=entity_texts1.keys(),
    columns=entity_texts2.keys(),
)

# Function to match entities based on similarity threshold
def match_entities(similarity_df, threshold):
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
    df_similarity, threshold
)

# End timer for the algorithm
algorithm_end_time = time.time()
algorithm_runtime = algorithm_end_time - algorithm_start_time

print(f"\n--- Algorithm Runtime ---")
print(f"Total time for deduplication: {algorithm_runtime:.4f} seconds")


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

file_path = "matches/example_matches.json"

with open(file_path, "w") as f:
    json.dump(final_result, f, indent=4)


# --- Evaluation starts here ---

# 1. Load the Golden Standard
try:
    golden_standard_df = pd.read_csv('example_data/updated_golden_standard_duplicates.csv')
except FileNotFoundError:
    print("Error: Ground truth not found Exiting.")
    exit()
except Exception as e:
    print(f"Error loading Ground truth: {e}. Exiting.")
    exit()

# Calculate and print entity-level P/R/F1 scores
# This requires 'original_entity_uri' and 'varied_entity_uri' in golden_standard_df
if 'golden_standard_df' in locals(): # Check if golden_standard_df was loaded
    calculate_entity_level_metrics(matched_entities, golden_standard_df)
else:
    print("Golden standard not loaded, skipping entity-level P/R/F1 calculation.")

# 2. Define Field Mapping
field_to_predicate_map = {
    "personName": "name", "birthDate": "birthDate", "knowsLanguage": "knowsLanguage", "gender": "gender",
    "email": "email", "jobTitle": "jobTitle",
    "city": "addressLocality", "postalCode": "postalCode", "country": "addressCountry", "text": "streetAddress",
    "healthcareOrganizationName": "name", "serviceDepartmentName": "name"
}

# 3. Define path to generated match file
file_path = "matches/example_matches.json"

# 4. Create a results DataFrame for field-level evaluation
results_df = pd.DataFrame()

# Create a dummy results dataframe to use with print_detailed_statistics 
# This is a simplified approach since we don't have analyze_match_results function
try:
    # Try to load matches file
    with open(file_path, 'r') as f:
        matches_data = json.load(f)
    
    print(f"\nSuccessfully loaded {len(matches_data)} entity matches from {file_path}")
    
    # Print details about the matches for reference
    print(f"Number of matched entity pairs: {len(matches_data)}")
    
    # Get field-level statistics directly using the available function
    stats_summary, variation_comparison = print_detailed_statistics(
        results_df,  # Empty DataFrame as we don't have field-level analysis
        golden_standard_df, 
        "Graph-Text Hybrid (Node2Vec)"
    )
    
    # Store statistics in a JSON file
    collected_detailed_stats = {
        "Graph-Text Hybrid (Node2Vec)": {
            'summary_statistics': stats_summary,
            'variation_analysis': variation_comparison.to_dict() # Convert DataFrame to dict for JSON serialization
        }
    }
    
    # Save collected detailed statistics to a JSON file
    output_stats_file = 'results/hybrid_node2vec_statistics.json'
    with open(output_stats_file, 'w') as f:
        json.dump(collected_detailed_stats, f, indent=4)
    print(f"\n--- Statistics saved to {output_stats_file} ---")

except Exception as e:
    print(f"Error processing match results: {e}")

print("\n--- Evaluation Script Finished ---")
