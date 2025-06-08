from node2vec import Node2Vec
import networkx as nx
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np
import rdflib

def rdf_to_nx(graph):
    G = nx.Graph()
    for s, p, o in graph:
        if isinstance(s, (str, int)) and isinstance(o, (str, int)):
            G.add_edge(str(s), str(o), predicate=str(p))
    return G

def get_graph_embeddings_Node2vec(graph, dimensions=384):
    G_nx = rdf_to_nx(graph)
    node2vec = Node2Vec(G_nx, dimensions=dimensions, walk_length=10, num_walks=60, workers=1)
    model = node2vec.fit()
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    return embeddings

def get_hybrid_vectorNode2vec(entity, text_vector, graph_embeddings, alpha=0.5, text_dim=384):
    """Return a single hybrid vector for one entity."""
    graph_vec = graph_embeddings.get(str(entity), np.zeros(text_dim))
    text_vec = np.array(text_vector.cpu().numpy()).flatten()
    graph_vec = np.array(graph_vec).flatten()
    return alpha * text_vec + (1 - alpha) * graph_vec

def get_hybrid_vectorsNode2vec(entities, text_vectors, graph_embeddings, alpha=0.5, text_dim=384):
    """Return an array of hybrid vectors for a list of entities and their text vectors."""
    return np.array([
        get_hybrid_vectorNode2vec(e, t, graph_embeddings, alpha=alpha, text_dim=text_dim)
        for e, t in zip(entities, text_vectors)
    ])



def get_graph_embeddings_PyKEEN(graph, model, dimensions=384, num_epochs=60):
    triples = [
        (str(s), str(p), str(o))
        for s, p, o in graph
        if not isinstance(s, rdflib.BNode) and not isinstance(o, rdflib.BNode)
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
