import pandas as pd
try:
    df_rope = pd.read_csv('results/icf/monash/results.csv')
    df_nope = pd.read_csv('results/icf/monash_nope/results.csv')
    
    # Merge on dataset and metric
    merged = pd.merge(
        df_rope[['dataset', 'metric', 'value']], 
        df_nope[['dataset', 'metric', 'value']], 
        on=['dataset', 'metric'], 
        suffixes=('_rope', '_nope')
    )
    
    # Calculate difference
    merged['diff (nope - rope)'] = merged['value_nope'] - merged['value_rope']
    
    print("MONASH RESULTS (RoPE vs NoPE)")
    print("===============================")
    for metric in ['mae', 'mase', 'smape']:
        print(f"\n--- Metric: {metric.upper()} ---")
        sub = merged[merged['metric'] == metric].copy()
        
        # Lower is better, so if diff > 0, NoPE is worse
        sub['better_model'] = sub['diff (nope - rope)'].apply(lambda x: 'RoPE' if x > 0 else 'NoPE')
        print(sub[['dataset', 'value_rope', 'value_nope', 'better_model']].to_string(index=False))

except Exception as e:
    print(f"Error: {e}")
