import shap
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import build_cnn  # Or load

# Load model and data (reuse from train)
model = tf.keras.models.load_model('models/best_cotton_model.h5')
# Assume X_train, X_test from train.py; load if needed
# For demo: background = X_train[:100].astype('float32')
# test_sample = X_test[0:1].astype('float32')

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_sample)

# Summary plot (importance across samples)
shap.summary_plot(shap_values[1], test_sample, feature_names=['A', 'C', 'G', 'T'], show=False)
plt.savefig('results/shap_summary.png')
plt.show()

# Force plot for one sequence (motif-like)
shap.force_plot(explainer.expected_value[1], shap_values[1][0], test_sample[0], matplotlib=True, show=False)
plt.savefig('results/shap_force.png')
print("SHAP plots saved: Highlights important bases/motifs (e.g., UTR regions).")