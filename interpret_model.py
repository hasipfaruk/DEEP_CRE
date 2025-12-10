import shap
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import SequenceLoader

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load model and data
model = tf.keras.models.load_model('models/best_cotton_model.h5')
tpm_df = pd.read_csv('data/tpm_counts.csv', index_col=0)
loader = SequenceLoader('data/genome.fasta', 'data/genes.gtf')
X, _, ids = loader.extract_sequences(tpm_df)

# Use subset for background and test
# Background: first 100 samples for SHAP background
# Test: next 50 samples for interpretation
background = X[:100].astype('float32')
test_sample = X[100:150].astype('float32')

print(f"Using {len(background)} samples for background and {len(test_sample)} samples for interpretation")

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_sample)

# Summary plot (importance across samples)
# For binary classification, shap_values is a list with one element
shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
shap.summary_plot(shap_vals, test_sample, feature_names=['A', 'C', 'G', 'T'], show=False)
plt.savefig('results/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Force plot for one sequence (motif-like)
expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
shap.force_plot(expected_val, shap_vals[0], test_sample[0], matplotlib=True, show=False)
plt.savefig('results/shap_force.png', dpi=300, bbox_inches='tight')
plt.close()

print("SHAP plots saved: Highlights important bases/motifs (e.g., UTR regions).")