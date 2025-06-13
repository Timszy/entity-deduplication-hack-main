# modular_methods/deduplication_pipeline.py

import json
import numpy as np
import torch
import torch.nn.functional as F
from modular_methods.graphToText_utils import get_entity_texts, get_literals_for_entities, group_by_type, traverse_graph_and_get_literals
from modular_methods.similarity_utils import compute_cosine_similarity, match_entities, Levenshtein_filter
from modular_methods.embedding_utils import get_hybrid_vectors

def deduplicate_graphs(
    phkg_graph,
    skg_graph,
    embedding_model,
    graph_embeddings=None,
    use_hybrid = False,
    alpha=0.5,
    text_dim=384,
    threshold=0.7,
    top_k=2,
    filter_literals=True
):
    # 1. Extract entity texts and group
    entity_texts1 = get_entity_texts(phkg_graph)
    entity_texts2 = get_entity_texts(skg_graph)
    grouped1 = group_by_type(entity_texts1)
    grouped2 = group_by_type(entity_texts2)

    # 2. Compute similarities by type
    all_matches = []
    for typ in set(grouped1) & set(grouped2):
        ids1, texts1 = zip(*grouped1[typ])
        ids2, texts2 = zip(*grouped2[typ])
        emb1 = embedding_model.encode(texts1, convert_to_tensor=True)
        emb2 = embedding_model.encode(texts2, convert_to_tensor=True)

        # Hybrid vector logic
        if use_hybrid and graph_embeddings is not None:
            
            hybrid_vecs1 = get_hybrid_vectors(ids1, emb1, graph_embeddings, alpha=alpha, text_dim=text_dim)
            hybrid_vecs2 = get_hybrid_vectors(ids2, emb2, graph_embeddings, alpha=alpha, text_dim=text_dim)
            emb1 = torch.tensor(hybrid_vecs1)
            emb2 = torch.tensor(hybrid_vecs2)
        
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        sim_matrix = compute_cosine_similarity(emb1, emb2)
        matches = match_entities(sim_matrix, ids1, ids2, threshold=threshold, top_k=top_k)
        all_matches.extend(matches)

    print(f"Total matches found: {len(all_matches)}")

    # 3. (Optional) Literal-based filtering
    if filter_literals:
        entities1 = set(ent1 for ent1, _, _ in all_matches)
        entities2 = set(ent2 for _, ent2, _ in all_matches)

        literals1 = get_literals_for_entities(phkg_graph, entities1)
        literals2 = get_literals_for_entities(skg_graph, entities2)
        filtered = Levenshtein_filter(all_matches, literals1, literals2)
        print(f"Filtered matches after literal check: {len(filtered)}/ {len(all_matches)}")
        

        return filtered
    # 4. Prepare final results
    else:
        return all_matches




def save_matches(matches, filename):
    with open(filename, "w") as f:
        json.dump(matches, f, indent=2)
