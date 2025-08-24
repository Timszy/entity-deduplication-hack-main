from node2vec import Node2Vec
from karateclub import NetMF
import networkx as nx
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import NodePiece
import numpy as np
import rdflib
from rdflib.term import URIRef

# from pyrdf2vec import RDF2VecTransformer
# from pyrdf2vec.embedders import Word2Vec
# from pyrdf2vec.graphs import KG
# from pyrdf2vec.walkers import RandomWalker

def rdf_to_nx_old(graph):
    G = nx.Graph()
    for s, p, o in graph:
        if isinstance(s, (str, int)) and isinstance(o, (str, int)):
            G.add_edge(str(s), str(o), predicate=str(p))
    return G


def rdf_to_nx(graph):
    G = nx.Graph()
    for s, p, o in graph:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            G.add_edge(str(s), str(o), predicate=str(p))
    return G

def get_graph_embeddings_Node2vec(graph, dimensions=384):
    G_nx = rdf_to_nx(graph)
    node2vec = Node2Vec(G_nx, dimensions=dimensions, walk_length=10, num_walks=100, workers=1)
    model = node2vec.fit()
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    return embeddings

def get_hybrid_vector(entity, text_vector, graph_embeddings, alpha=0.5, text_dim=384):
    """Return a single hybrid vector for one entity."""
    graph_vec = graph_embeddings.get(str(entity), np.zeros(text_dim))
    text_vec = np.array(text_vector.cpu().numpy()).flatten()
    graph_vec = np.array(graph_vec).flatten()
    return alpha * text_vec + (1 - alpha) * graph_vec

def get_hybrid_vectors(entities, text_vectors, graph_embeddings, alpha=0.5, text_dim=384):
    """Return an array of hybrid vectors for a list of entities and their text vectors."""
    return np.array([
        get_hybrid_vector(e, t, graph_embeddings, alpha=alpha, text_dim=text_dim)
        for e, t in zip(entities, text_vectors)
    ])




def get_graph_embeddings_NetMF(graph, dimensions=384):
    # Convert RDFLib graph to NetworkX
    G_nx = rdf_to_nx(graph)
    # Relabel nodes as 0...N-1 integers and keep mapping
    node_list = list(G_nx.nodes())
    mapping = {node: idx for idx, node in enumerate(node_list)}
    inv_mapping = {idx: node for node, idx in mapping.items()}
    G_int = nx.relabel_nodes(G_nx, mapping)
    
    model = NetMF(dimensions=dimensions)
    model.fit(G_int)
    embeddings = model.get_embedding()
    
    # Critical: Use G_int.nodes() order to align embeddings with indices
    ordered_nodes = list(G_int.nodes())
    return {inv_mapping[idx]: embeddings[i] for i, idx in enumerate(ordered_nodes)}



def get_graph_embeddings_PyKEEN(graph, model, dimensions=384, num_epochs=100):
    triples = [
    (str(s), str(p), str(o))
    for s, p, o in graph
    if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef)
    ]
    triples_array = np.array(triples, dtype=str)
    triples_factory = TriplesFactory.from_labeled_triples(triples_array)
    training_factory, testing_factory = triples_factory.split([0.8, 0.2], random_state=69)
    result = pipeline(
        training=training_factory,
        testing=testing_factory,
        model=model,
        model_kwargs=dict(embedding_dim=dimensions),
        training_loop='slcwa',
        training_kwargs=dict(num_epochs=num_epochs),
        evaluator_kwargs=dict(filtered=True),
        random_seed=69,
    )
    entity_to_id = triples_factory.entity_to_id
    model_graph = result.model
    graph_embedding_matrix = model_graph.entity_representations[0]().detach().cpu().numpy().real
    graph_embeddings = {e: graph_embedding_matrix[i] for e, i in entity_to_id.items()}
    return graph_embeddings

def get_graph_embeddings_PyKEEN_nodepiece(graph, dimensions=384, num_epochs=60):
    triples = [ (str(s), str(p), str(o))
        for s, p, o in graph
        if not isinstance(s, rdflib.BNode) and not isinstance(o, rdflib.BNode)]
    triples_array = np.array(triples, dtype=str)
    triples_factory = TriplesFactory.from_labeled_triples(
        triples_array,
        create_inverse_triples=True
    )
    training_factory, testing_factory = triples_factory.split([0.8, 0.2], random_state=69)

    model = NodePiece(
        triples_factory=training_factory,
        tokenizers=["AnchorTokenizer", "RelationTokenizer"],
        num_tokens=[20, 12],
        embedding_dim=dimensions,
        interaction="DistMult",
    )

    result = pipeline(
        training=training_factory,
        testing=testing_factory,
        model=model,
        training_loop="slcwa",
        training_kwargs=dict(num_epochs=num_epochs),
        evaluator_kwargs=dict(filtered=True),
        random_seed=69,
    )

    entity_to_id = triples_factory.entity_to_id
    graph_embedding_matrix = result.model.entity_representations[0]().detach().cpu().numpy().real
    return {e: graph_embedding_matrix[i] for e, i in entity_to_id.items()}
