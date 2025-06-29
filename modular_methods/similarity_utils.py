# modular_methods/similarity_utils.py
import re
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


def get_acronym(s):
    """
    Extracts an acronym from a string, e.g., 'Delgado, Guerrero and Simpson Zorg' -> 'DGSZ'
    """
    words = re.findall(r'\b\w', s)
    return ''.join(words).upper()

def literal_based_threshold(n_literals):
    """
    Return a threshold based on the number of common literals.
    """
    thresholds = {1:0.4, 2: 0.55, 3: 0.7, 4: 0.8, 5: 0.85}
    return thresholds.get(n_literals, 0.85)  # default to 0.85 if out of range

def Levenshtein_filter_flag(matches, literals1, literals2, acronym_boost=0.95):
    """
    Post-process entity matches by comparing their predicates using Levenshtein and acronym matching.
    Threshold is adjusted based on the number of literals in each entity (from 1 to 5).
    """
    filtered = []
    for ent1, ent2, score in matches:
        preds1 = literals1.get(str(ent1), {})
        preds2 = literals2.get(str(ent2), {})
        common_preds = set(preds1.keys()) & set(preds2.keys())
        if not common_preds:
            continue
        sim_scores = []
        for p in common_preds:
            val1 = str(preds1[p]).lower()
            val2 = str(preds2[p]).lower()
            sim = normalized_levenshtein(val1, val2)
            # Acronym check
            acronym1 = get_acronym(val1)
            acronym2 = get_acronym(val2)
            if acronym1 == val2.replace(" ", "").upper() or acronym2 == val1.replace(" ", "").upper():
                sim = max(sim, acronym_boost)
            sim_scores.append(sim)
        avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        n_literals = len(common_preds)
        threshold = literal_based_threshold(n_literals)
        
        if avg_sim >= threshold:
            filtered.append((ent1, ent2, score, avg_sim, "pass"))
        elif avg_sim < threshold and n_literals < 3:
            filtered.append((ent1, ent2, score, avg_sim, "fail"))
    return filtered

def Levenshtein_filter(matches, literals1, literals2, filter=True, acronym_boost=0.95 ):
    """
    Post-process entity matches by comparing their predicates using Levenshtein and acronym matching.
    Threshold is adjusted based on the number of literals in each entity (from 1 to 5).
    """
    filtered = []
    for ent1, ent2, score in matches:
        preds1 = literals1.get(str(ent1), {})
        preds2 = literals2.get(str(ent2), {})
        common_preds = set(preds1.keys()) & set(preds2.keys())
        if not common_preds:
            continue
        sim_scores = []
        for p in common_preds:
            val1 = str(preds1[p]).lower()
            val2 = str(preds2[p]).lower()
            sim = normalized_levenshtein(val1, val2)
            # Acronym check
            acronym1 = get_acronym(val1)
            acronym2 = get_acronym(val2)
            if acronym1 == val2.replace(" ", "").upper() or acronym2 == val1.replace(" ", "").upper():
                sim = max(sim, acronym_boost)
            sim_scores.append(sim)
        avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        n_literals = len(common_preds)
        threshold = literal_based_threshold(n_literals)
        
        if filter:
            if avg_sim >= threshold:
                filtered.append((ent1, ent2, score, avg_sim, "pass"))
            elif avg_sim < threshold and n_literals < 3:
                filtered.append((ent1, ent2, score, avg_sim, "fail"))
        else:
            if avg_sim >= threshold:
                filtered.append((ent1, ent2, score, avg_sim, "pass"))
            elif avg_sim < threshold:
                filtered.append((ent1, ent2, score, avg_sim, "fail"))
    return filtered