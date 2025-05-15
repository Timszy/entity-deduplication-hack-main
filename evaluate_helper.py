import json
import pandas as pd

def analyze_match_results(match_file_path, golden_standard_df, field_to_predicate_map, match_source_name):
    """
    Analyzes a single match file against the golden standard.
    """
    print(f"\n--- Analyzing: {match_source_name} ({match_file_path}) ---")
    
    try:
        with open(match_file_path, 'r') as f:
            matches_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Match file not found at {match_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {match_file_path}")
        return None

    current_matched_results = []
    for match in matches_data:
        try:
            entity1 = match['entities'][0]['entity1']
            entity2 = match['entities'][1]['entity2']
            
            for predicate1 in entity1.get('predicates', []): # Use .get for safety
                predicate_suffix = predicate1['predicate'].split('/')[-1]
                object1 = predicate1['object']
                
                for predicate2 in entity2.get('predicates', []): # Use .get for safety
                    object2 = predicate2['object']
                    
                    condition = (
                        ((golden_standard_df['field_name'] == predicate_suffix) | 
                         (golden_standard_df['field_name'].map(lambda x: field_to_predicate_map.get(x)) == predicate_suffix)) &
                        (golden_standard_df['original_value'] == object1) &
                        (golden_standard_df['varied_value'] == object2)
                    )
                    
                    # Get all rows from golden_standard that satisfy the condition
                    matching_gs_rows = golden_standard_df[condition]
                    
                    if not matching_gs_rows.empty:
                        # If multiple golden standard rows match, log each one.
                        # Typically, we expect one, but this handles edge cases.
                        for _, gs_row in matching_gs_rows.iterrows():
                            current_matched_results.append({
                                'entity1_subject': entity1['subject'],
                                'entity2_subject': entity2['subject'],
                                'predicate': predicate_suffix,
                                'original_value': object1,
                                'varied_value': object2,
                                'similarity_score': match['similarity_score'],
                                'duplication_type': match['duplication_type'],
                                'variation_type': gs_row['variation_type'],
                                'match_source': match_source_name
                            })
        except KeyError as e:
            print(f"Warning: Skipping a match due to missing key: {e} in {match_source_name}")
            continue
        except Exception as e:
            print(f"Warning: An unexpected error occurred while processing a match: {e} in {match_source_name}")
            continue
            
    if not current_matched_results:
        print(f"No confirmed field-level matches found for {match_source_name}.")
        return pd.DataFrame() # Return empty DataFrame

    return pd.DataFrame(current_matched_results)

def print_detailed_statistics(results_df, golden_standard_df, source_name):
    """
    Prints detailed statistics for the matched results.
    """
    print(f"\n--- Statistics for {source_name} ---")
    
    total_golden_field_variations = len(golden_standard_df) # Total field variations in golden standard
    
    if results_df.empty:
        print(f"No confirmed field-level matches to analyze for {source_name}.")
        print(f"Total Field Variations in Golden Standard: {total_golden_field_variations}")
        if total_golden_field_variations > 0:
            variation_type_counts_golden = golden_standard_df['variation_type'].value_counts()
            print("\nGolden Standard Variation Type Counts (Field-Level):")
            print(variation_type_counts_golden)
        return

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
