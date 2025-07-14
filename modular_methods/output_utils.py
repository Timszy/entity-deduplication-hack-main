# modular_methods/output_utils.py

from modular_methods.graphToText_utils import traverse_graph_and_get_literals

def build_final_result(matches, graph1, graph2, graph1_name="phkg_graph", graph2_name="g2"):
    final_result = []
    for match in matches:
        # match is (ent1, ent2, score) or (ent1, ent2, score, avg_literal_similarity)
        ent1, ent2, score, avg_sim, status, true_duplicate = match # status is not included in the match tuple
        entity1_literals = traverse_graph_and_get_literals(graph1, ent1)
        entity2_literals = traverse_graph_and_get_literals(graph2, ent2)
        score_str = str(float(score))

        entity1_predicates = entity1_literals.get(str(ent1), {})
        entity2_predicates = entity2_literals.get(str(ent2), {})

        all_predicates = sorted(set(entity1_predicates.keys()) | set(entity2_predicates.keys()))

        entity1_details = {
            "from": graph1_name,
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
            "from": graph2_name,
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

        # Conditional logic for avg_literal_similarity
        result_entry = {
            "entities": [
                {"entity1": entity1_details},
                {"entity2": entity2_details}
            ],
            "similarity_score": score_str,
        }

        # If avg_literal_similarity present, include it and optionally use it for duplication type
        if avg_sim is not None:
            avg_literal_similarity = avg_sim
            result_entry["status"] = status
            result_entry["avg_literal_similarity"] = str(float(avg_literal_similarity))
            # You can use literal similarity instead of embedding similarity for duplication type
            duplication_type = (
                #"flagged" if status == "flagged" else
                "true_duplicate" if true_duplicate == 'exact' else
                "near-exact" if float(avg_literal_similarity) >= 0.9 else
                "similar" if float(avg_literal_similarity) >= 0.7 else
                "conflict"
            )
        else:
            duplication_type = (
                "true_duplicate" if duplication_type == 'exact' else
                "near-exact" if float(score) >= 0.9 else
                "similar" if float(score) >= 0.7 else
                "conflict"
            )
        result_entry["duplication_type"] = duplication_type

        final_result.append(result_entry)
    return final_result

