import json
import pandas as pd
import rdflib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesNumericLiteralsFactory

# =============================
# ========= Configs ===========
# =============================
TEXT_MODEL = "all-MiniLM-L6-v2"
TEXT_DIM = 384
THRESHOLD = 0.5
NUM_EPOCHS = 80
GRAPH_MODEL = "DistMultLiteral"
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
# =============================
# Load graph
def load_graph():
    g1 = rdflib.Graph()
    g2 = rdflib.Graph()
    master_graph = rdflib.Graph()
    g1.parse("data/healthcare_graph_original_v2.ttl")
    g2.parse("data/prog_data/healthcare_graph_var_only.ttl")
    master_graph.parse("data/master_data.ttl")
    phkg_graph = g1 + master_graph
    combined_graph = phkg_graph + g2
    return combined_graph, phkg_graph, g1, g2, master_graph

# Extract triples for knowledge graph embeddings
def extract_triples(graph):
    """Extract triples from an RDF graph, skipping blank nodes."""
    triples = []
    for s, p, o in graph:
        if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
            continue
        triples.append((str(s), str(p), str(o)))
    return triples


# =============================
# ======= Handling text =======
# =============================
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
        #print(f"Entity {s} with text: {text}")
    return entity_texts


# =============================
# ==== Code for the model =====
# =============================
def run(triples_array, text_embeddings):
    # Map entities to IDs
    entities = set(triples_array[:, 0]).union(set(triples_array[:, 2]))
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}

    # Create numeric triples: (entity, feature_i, value)
    numeric_triples = []
    for entity, vector in text_embeddings.items():
        if entity not in entity_to_id:
            continue
        for i, val in enumerate(vector):
            predicate = f"hasTextFeature_{i}"
            numeric_triples.append((entity, predicate, str(val)))

    numeric_triples_array = np.array(numeric_triples, dtype=str)

    # Build triples factory with numeric literals
    triples_factory = TriplesNumericLiteralsFactory.from_labeled_triples(
        triples=triples_array,
        numeric_triples=numeric_triples_array,
        entity_to_id=entity_to_id
    )

    # Train/test split and model pipeline
    training_factory, testing_factory = triples_factory.split([0.8, 0.2])
    
    # Call pipeline
    result = pipeline(
        model=GRAPH_MODEL,
        model_kwargs=dict(
            embedding_dim=TEXT_DIM,
        ),
        training=training_factory,
        testing=testing_factory,
        training_loop='slcwa',
        training_kwargs=dict(num_epochs=NUM_EPOCHS),
        evaluator_kwargs=dict(filtered=True),
    )
    return result, entity_to_id
    
    
# =============================
# ========= Matching ==========
# =============================
def match_entities(ids1, ids2, entity_to_id, embedding_matrix):
    vecs1 = [embedding_matrix[entity_to_id[str(e)]] for e in ids1 if str(e) in entity_to_id]
    vecs2 = [embedding_matrix[entity_to_id[str(e)]] for e in ids2 if str(e) in entity_to_id]
    df_similarity = pd.DataFrame(
        cosine_similarity(vecs1, vecs2),
        index=[e for e in ids1 if str(e) in entity_to_id],
        columns=[e for e in ids2 if str(e) in entity_to_id],
    )
    matches = []
    for idx in df_similarity.index:
        max_sim = df_similarity.loc[idx].max()
        if max_sim >= THRESHOLD:
            best_match = df_similarity.loc[idx].idxmax()
            matches.append((idx, best_match, max_sim))
    return matches


# =============================
# ========== Main =============
# =============================
def main():
    combined_graph, phkg_graph, g1, g2, master_graph = load_graph()
    
    entity_texts1 = get_entity_texts(phkg_graph)
    entity_texts2 = get_entity_texts(g2)
    ids1 = list(entity_texts1.keys())
    ids2 = list(entity_texts2.keys())

    all_texts = {**entity_texts1, **entity_texts2}
    model = SentenceTransformer(TEXT_MODEL)
    text_embeddings = {
        str(e): model.encode(text, convert_to_numpy=True)
        for e, text in all_texts.items()
    }

    triples = extract_triples(combined_graph)
    triples_array = np.array(triples, dtype=str)

    result, entity_to_id = run(triples_array, text_embeddings)
    embedding_matrix = result.model.entity_representations[0]().detach().cpu().numpy().real

    matches = match_entities(ids1, ids2, entity_to_id, embedding_matrix)
    match_json = [{"entity1": str(e1), "entity2": str(e2), "score": float(score)} for e1, e2, score in matches]

    with open("DistLitmatches.json", "w") as f:
        json.dump(match_json, f, indent=4)

if __name__ == "__main__":
    main()

