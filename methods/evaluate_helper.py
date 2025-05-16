import json
import pandas as pd

def calculate_entity_level_metrics(algorithm_matches, golden_standard_df):
    """
    Calculates and prints precision, recall, and F1 score for entity-level matches.

    Args:
        algorithm_matches (list): A list of tuples (ent1, ent2, score) from the matching algorithm.
                                  ent1 and ent2 are expected to be rdflib.URIRef or string URIs.
        golden_standard_df (pd.DataFrame): DataFrame of the golden standard.
                                           Expected to have 'original_entity_uri' and 'varied_entity_uri' columns.
    """
    print("\n--- Entity-Level Evaluation Metrics ---")

    if 'original_id' not in golden_standard_df.columns or \
       'duplicate_id' not in golden_standard_df.columns:
        print("Error: Golden standard DataFrame must contain 'original_entity_uri' and 'varied_entity_uri' columns.")
        print("Skipping entity-level P/R/F1 calculation.")
        return

    # Extract predicted pairs from algorithm_matches
    # Ensuring canonical form (sorted tuple of URIs) to count pairs uniquely
    predicted_pairs = set()
    for ent1, ent2, _ in algorithm_matches:
        uri1, uri2 = str(ent1), str(ent2)
        predicted_pairs.add(tuple(sorted((uri1, uri2))))

    # Extract true pairs from golden_standard_df
    # Ensuring canonical form
    true_pairs = set()
    for _, row in golden_standard_df.iterrows():
        uri1 = str(row['original_id'])
        uri2 = str(row['duplicate_id'])
        if pd.notna(uri1) and pd.notna(uri2) and uri1 and uri2 : # Ensure URIs are not NaN or empty
            true_pairs.add(tuple(sorted((uri1, uri2))))

    if not true_pairs:
        print("No valid entity pairs found in the golden standard based on URI columns.")
        if not predicted_pairs:
            print("No pairs predicted by the algorithm.")
        else:
            print(f"Algorithm predicted {len(predicted_pairs)} pairs, but cannot compare without golden standard pairs.")
        return

    tp = len(predicted_pairs.intersection(true_pairs))
    fp = len(predicted_pairs.difference(true_pairs))
    fn = len(true_pairs.difference(predicted_pairs))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Total Predicted Pairs: {len(predicted_pairs)}")
    print(f"Total True Pairs in Golden Standard: {len(true_pairs)}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")


def print_detailed_statistics(results_df, golden_standard_df, source_name):
    """
    Prints detailed statistics for the matched results and returns them.
    """
    print(f"\n--- Statistics for {source_name} ---")
    
    total_golden_field_variations = len(golden_standard_df) # Total field variations in golden standard
    
    if results_df.empty:
        print(f"No confirmed field-level matches to analyze for {source_name}.")
        print(f"Total Field Variations in Golden Standard: {total_golden_field_variations}")
        stats_summary = {
            'Total Field Variations in Golden Standard': total_golden_field_variations,
            'Total Confirmed Matched Field Variations': 0,
            'Overall Percentage of Golden Field Variations Matched (%)': 0.0
        }
        variation_comparison = pd.DataFrame(columns=['Golden Standard Count', 'Confirmed Matched Count', 'Matched (%)'])
        if total_golden_field_variations > 0:
            variation_type_counts_golden = golden_standard_df['variation_type'].value_counts()
            print("\nGolden Standard Variation Type Counts (Field-Level):")
            print(variation_type_counts_golden)
            variation_comparison['Golden Standard Count'] = variation_type_counts_golden
            variation_comparison = variation_comparison.fillna(0)

        return stats_summary, variation_comparison

    # These are confirmed field-level matches based on the golden standard
    variation_type_counts_golden = golden_standard_df['variation_type'].value_counts()
    variation_type_counts_matched = results_df['variation_type'].value_counts()

    total_confirmed_matched_field_variations = variation_type_counts_matched.sum()
    
    percentage_matched = 0
    if total_golden_field_variations > 0:
        percentage_matched = (total_confirmed_matched_field_variations / total_golden_field_variations) * 100
    
    variation_comparison = pd.DataFrame({
        'Golden Standard Count': variation_type_counts_golden,
        'Confirmed Matched Count': variation_type_counts_matched  # Renamed for clarity
    }).fillna(0).astype(int)

    variation_comparison['Matched (%)'] = 0.0
    for index, row_data in variation_comparison.iterrows():
        gs_count = row_data['Golden Standard Count']
        matched_count = row_data['Confirmed Matched Count']
        if gs_count > 0:
            variation_comparison.loc[index, 'Matched (%)'] = (matched_count / gs_count) * 100
        elif matched_count > 0 : # Matched but no corresponding golden standard type (should not happen with current logic)
             variation_comparison.loc[index, 'Matched (%)'] = float('inf')


    stats_summary = {
        'Total Field Variations in Golden Standard': total_golden_field_variations,
        'Total Confirmed Matched Field Variations': total_confirmed_matched_field_variations,
        'Overall Percentage of Golden Field Variations Matched (%)': percentage_matched
    }

    print("\nField-Level Matching Statistics:")
    for key, value in stats_summary.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nVariation Type Analysis (Field-Level):")
    print(variation_comparison)
    print(f"Number of unique variation types found in {source_name} confirmed matches: {len(variation_type_counts_matched)}")
    return stats_summary, variation_comparison
