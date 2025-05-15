import json
import pandas as pd
import rdflib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Load the RDF graphs
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

# Replace 'graph1.rdf' and 'graph2.rdf' with the paths to your RDF files
g1.parse("/Users/nguyenhoanghai/Documents/GitHub/entity-deduplication-hack-main/data/healthcare_graph_original.ttl")
g2.parse("/Users/nguyenhoanghai/Documents/GitHub/entity-deduplication-hack-main/data/healthcare_graph_replaced.ttl")
master_graph.parse("/Users/nguyenhoanghai/Documents/GitHub/entity-deduplication-hack-main/data/master_data.ttl")

phkg_graph = g1 + master_graph

alpha = 0.0 # You can change this value to weight the text embedding (0.0 = is graph only)
text_dim = 384 # Dim for the all-MiniLM-L6-v2
threshold = 0.5
num_epochs = 500 # You can change it as you like

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
# This method is needed for the TransE, we are not converting the graph into networkx anymore, since this embedding is used for knowledge graphs 
# Use triplets for relational information, like two nodes and the edge between them
def extract_triples(graph):
    """Extract triples from an RDF graph, skipping blank nodes."""
    triples = []
    for s, p, o in graph:
        if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
            continue
        triples.append((str(s), str(p), str(o)))
    return triples

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
# Get triplets
triples = extract_triples(combined_graph)

# Convert to a proper NumPy array with dtype=str and shape (n, 3)
triples_array = np.array(triples, dtype=str).reshape(-1, 3)

# Create the triples factory
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

# Split into training and testing (e.g., 80/20 split)
training_factory, testing_factory = triples_factory.split([0.8, 0.2])
result = pipeline(
    training=training_factory,
    testing=testing_factory,
    validation=triples_factory,
    model='TransE',
    model_kwargs=dict(embedding_dim=text_dim),
    training_loop='slcwa',  
    training_kwargs=dict(num_epochs=num_epochs, use_tqdm_batch=False),
    evaluator_kwargs=dict(filtered=True),
)

# Get all entity embeddings (as numpy array)
entity_embeddings_tensor = result.model.entity_representations[0]().detach().cpu()
embedding_matrix = entity_embeddings_tensor.numpy()

# Get entity -> ID mapping
entity_to_id = triples_factory.entity_to_id

print("Entities from ids1 in graph embeddings:", sum(str(e) in entity_to_id for e in ids1))
print("Entities from ids2 in graph embeddings:", sum(str(e) in entity_to_id for e in ids2))

# Build final embedding dictionary
graph_embeddings = {
    entity: embedding_matrix[idx]
    for entity, idx in entity_to_id.items()
}

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

final_result = []

# Display the matches
for ent1, ent2, score in matched_entities:
    final_result.append(
        {
            "entity1": ent1,
            "entity2": ent2,
            "score": str(score),
        }
    )


with open("matches_transE.json", "w") as f:
    json.dump(final_result, f, indent=4)

