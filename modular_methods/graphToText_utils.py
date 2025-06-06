# modular_methods/graph_utils.py

import rdflib
from rdflib.namespace import RDF
from urllib.parse import urlparse
import re

WEAK_PREDICATES = {"schema:identifier"}

def camel_to_title(s: str) -> str:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return spaced.title()

def get_human_label(curie: str) -> str:
    if ":" in curie:
        _, local = curie.split(":", 1)
    else:
        local = curie
    if local.startswith("[") and local.endswith("]"):
        local = local[1:-1]
    return camel_to_title(local)

def get_prefixed_predicate(uri: str) -> str:
    if uri.startswith(("http://schema.org/", "https://schema.org/")):
        return "schema:" + uri.split("/")[-1]
    else:
        frag = urlparse(uri).fragment
        return frag if frag else uri.split("/")[-1]

def traverse_graph_and_get_literals(graph, subject) -> dict:
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

def get_literals_for_entities(graph, entities):
    return {str(e): traverse_graph_and_get_literals(graph, e).get(str(e), {}) for e in entities}

def create_text_from_literals(subject_uri, literals, graph):
    parts = []
    for o in graph.objects(rdflib.URIRef(subject_uri), RDF.type):
        curie = get_prefixed_predicate(str(o))
        human_type = get_human_label(curie)
        parts.append(f"Type: {human_type}.")
        break
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

def get_entity_texts_dist(graph): #Distmult can't handle dimensions so we return only text
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
            texts[s] = (text)
    return texts

def group_by_type(entity_dict):
    grouped = {}
    for entity, (text, typ) in entity_dict.items():
        grouped.setdefault(typ, []).append((entity, text))
    return grouped
