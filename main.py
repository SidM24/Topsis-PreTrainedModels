import pandas as pd
import numpy as np


def topsis(data, weights, impacts):
    # Normalize the data
    normalized_data = data / np.sqrt((data ** 2).sum(axis=0))

    # Multiply the normalized data by the weights
    weighted_normalized_data = normalized_data * weights

    # Calculate the ideal and negative-ideal solutions
    ideal_best = np.max(weighted_normalized_data, axis=0)
    ideal_worst = np.min(weighted_normalized_data, axis=0)

    # Calculate the separation measures
    separation_best = np.sqrt(((weighted_normalized_data - ideal_best) ** 2).sum(axis=1))
    separation_worst = np.sqrt(((weighted_normalized_data - ideal_worst) ** 2).sum(axis=1))

    # Calculate the TOPSIS score
    topsis_score = separation_worst / (separation_best + separation_worst)

    return topsis_score


# Read data from CSV
data = pd.read_csv('result.csv')

# Define weights for each criterion
weights = np.array([1, 1, 1, 1])

# Define impacts for each criterion (1 for benefit, -1 for cost)
impacts = np.array([1, 1, 1, 1])

# Extract relevant columns for TOPSIS
topsis_data = data[['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Content_Overlap']]

# Apply TOPSIS
data['Topsis Score'] = topsis(topsis_data.values, weights, impacts)

# Rank the models based on TOPSIS score
data['Final Rank'] = data['Topsis Score'].rank(ascending=False)

# Add row names as a new column
data['Model'] = ['t5-small-booksum', 'distlbart-cnn', 'Falconsai', 'bart-factbook', 'bart-large-xsum']

# Reorder columns to have 'Model' as the first column
data = data[['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Content_Overlap', 'Topsis Score', 'Final Rank']]

# Save the updated data to a new CSV file
data.to_csv('output_with_topsis.csv', index=False)
