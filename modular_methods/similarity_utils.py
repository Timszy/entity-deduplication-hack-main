# modular_methods/similarity_utils.py

import torch
import torch.nn.functional as F
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity between two sets of embeddings.
    emb1, emb2: torch.Tensor or np.ndarray, shape (n_samples, n_features)
    Returns a pandas DataFrame.
    """
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.cpu()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.cpu()
    sim_matrix = cosine_similarity(emb1, emb2)
    return sim_matrix

def match_entities(sim_matrix, ids1, ids2, threshold=0.7, top_k=2):
    """
    Given a similarity matrix and entity ids, return top matches above threshold.
    """
    df_sim = pd.DataFrame(sim_matrix, index=ids1, columns=ids2)
    matches = []
    for g2_entity in df_sim.columns:
        top_matches = df_sim[g2_entity].nlargest(top_k)
        for phkg_entity, sim in top_matches.items():
            if sim >= threshold:
                matches.append((phkg_entity, g2_entity, float(sim)))
    return matches

def normalized_levenshtein(a, b):
    """
    Return a similarity ratio between two strings using Levenshtein.
    """
    return difflib.SequenceMatcher(None, a, b).ratio()

def Levenshtein_filter(matches, literals1, literals2, threshold=0.63):
    """
    Post-process entity matches by comparing their predicates using Levenshtein.
    matches: list of (entity1, entity2, score)
    literals1/literals2: dict from entity URI to predicate:value dict
    Returns filtered matches.
    """
    filtered = []
    for ent1, ent2, score in matches:
        preds1 = literals1.get(str(ent1), {})
        preds2 = literals2.get(str(ent2), {})
        common_preds = set(preds1.keys()) & set(preds2.keys())
        if not common_preds:
            continue
        sim_scores = [
            normalized_levenshtein(str(preds1[p]).lower(), str(preds2[p]).lower())
            for p in common_preds
        ]
        avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        if avg_sim >= threshold:
            filtered.append((ent1, ent2, score, avg_sim))
    return filtered
