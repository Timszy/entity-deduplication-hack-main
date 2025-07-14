import rdflib
import numpy as np
import torch
from pykeen.triples import TriplesFactory
from pykeen.models.inductive import InductiveNodePieceGNN
from pykeen.losses import NSSALoss
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
import torch.nn.functional as F
from modular_methods.similarity_utils import compute_cosine_similarity, match_entities
from modular_methods.graphToText_utils import get_literals_for_entities
from modular_methods.dedup_pipeline import deduplicate_graphs, save_matches
from modular_methods.output_utils import build_final_result

### ---- 1. Load RDF graphs ----

main_graph = rdflib.Graph()
train_graph = rdflib.Graph()
test_graph = rdflib.Graph()

main_graph.parse("data/healthcare_graph_Main.ttl")
train_graph.parse("data/healthcare_graph_train.ttl")
test_graph.parse("data/healthcare_graph_replaced_high.ttl")

# Combine for inductive training/testing
train_combined = main_graph + train_graph
test_combined = main_graph + test_graph

### ---- 2. Convert graphs to triples arrays ----

def graph_to_triples(g):
    return [
        (str(s), str(p), str(o))
        for s, p, o in g
        if not isinstance(s, rdflib.BNode) and not isinstance(o, rdflib.BNode)
    ]

train_triples = np.array(graph_to_triples(train_combined))
test_triples = np.array(graph_to_triples(test_combined))
main_triples = np.array(graph_to_triples(main_graph))
test_only_triples = np.array(graph_to_triples(test_graph))

### ---- 3. Build TriplesFactory objects ----

tf_train = TriplesFactory.from_labeled_triples(train_triples, create_inverse_triples=True)
tf_test = TriplesFactory.from_labeled_triples(
    test_triples,
    entity_to_id=tf_train.entity_to_id,
    relation_to_id=tf_train.relation_to_id,
    create_inverse_triples=True
)
tf_main = TriplesFactory.from_labeled_triples(
    main_triples,
    entity_to_id=tf_train.entity_to_id,
    relation_to_id=tf_train.relation_to_id,
    create_inverse_triples=True
)
tf_test_only = TriplesFactory.from_labeled_triples(
    test_only_triples,
    entity_to_id=tf_train.entity_to_id,
    relation_to_id=tf_train.relation_to_id,
    create_inverse_triples=True
)

### ---- 4. Train InductiveNodePieceGNN ----

model = InductiveNodePieceGNN(
    triples_factory=tf_train,
    inference_factory=tf_test,
    num_tokens=12,
    aggregation="mlp",
    embedding_dim=128,
    interaction="DistMult",
    loss=NSSALoss(margin=15),
    random_seed=42,
).to("cuda" if torch.cuda.is_available() else "cpu")

optimizer = Adam(model.parameters(), lr=0.0005)

training_loop = SLCWATrainingLoop(
    triples_factory=tf_train,
    model=model,
    optimizer=optimizer,
    mode="training"
)

print("Training NodePiece...")
training_loop.train(tf_train, num_epochs=10)

### ---- 5. Extract entity embeddings ----

def extract_embeddings(model, triples_factory, mode="training"):
    model.mode = mode
    emb_array = model.entity_representations[0]().detach().cpu().numpy()
    entities = list(triples_factory.entity_to_id.keys())
    return {e: emb_array[i] for i, e in enumerate(entities)}

main_embeddings = extract_embeddings(model, tf_main, mode="training")
test_embeddings = extract_embeddings(model, tf_test_only, mode="testing")

# Make sure only common entities/types are compared (as your pipeline does)
entity_ids1 = list(main_embeddings.keys())
entity_ids2 = list(test_embeddings.keys())

emb1 = torch.tensor([main_embeddings[e] for e in entity_ids1])
emb2 = torch.tensor([test_embeddings[e] for e in entity_ids2])

emb1 = F.normalize(emb1, p=2, dim=1)
emb2 = F.normalize(emb2, p=2, dim=1)

sim_matrix = compute_cosine_similarity(emb1, emb2)
matches = match_entities(sim_matrix, entity_ids1, entity_ids2, threshold=0.7, top_k=5)

# Literal-based filtering (as in your pipeline)
literals1 = get_literals_for_entities(main_graph, entity_ids1)
literals2 = get_literals_for_entities(test_graph, entity_ids2)
from modular_methods.similarity_utils import Levenshtein_filter
filtered_matches = Levenshtein_filter(matches, literals1, literals2, filter=True)

### ---- 7. Format and save results ----

final_result = build_final_result(
    filtered_matches,
    main_graph,
    test_graph,
    graph1_name="MainGraph",
    graph2_name="TestGraph"
)
save_matches(final_result, "NodePiece_dedup_results.json")
print(f"Saved results to NodePiece_dedup_results.json")
