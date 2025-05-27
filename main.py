import json
import pandas as pd
import rdflib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the RDF graphs
g1 = rdflib.Graph()
g2 = rdflib.Graph()
master_graph = rdflib.Graph()

# Replace 'graph1.rdf' and 'graph2.rdf' with the paths to your RDF files
g1.parse("data/healthcare_graph_original_v2.ttl")
g2.parse("data/prog_data/healthcare_graph_var_only.ttl")
master_graph.parse("data/master_data.ttl")


phkg_graph = g1 + master_graph


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


def create_text_from_literals(
    literals: dict[str, dict[str, str]],
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

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
embeddings1 = model.encode(texts1, convert_to_tensor=True)
embeddings2 = model.encode(texts2, convert_to_tensor=True)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(
    embeddings1.cpu(), embeddings2.cpu()
)

# Convert similarity matrix to DataFrame for easier handling
df_similarity = pd.DataFrame(
    similarity_matrix,
    index=entity_texts1.keys(),
    columns=entity_texts2.keys(),
)


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
    df_similarity, threshold=0.7
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
    
with open("matchesHacky.json", "w") as f:
    json.dump(final_result, f, indent=4)

