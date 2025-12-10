import tensorflow as tf
import pandas as pd
from utils import SequenceLoader

model = tf.keras.models.load_model('models/best_cotton_model.h5')
tpm_df = pd.read_csv('data/tpm_counts.csv', index_col=0)
loader = SequenceLoader('data/genome.fasta', 'data/genes.gtf')
X, _, ids = loader.extract_sequences(tpm_df)  # Use full for preds

probs = model.predict(X, batch_size=32).flatten()
df_out = pd.DataFrame({'gene_id': ids, 'pred_high_prob': probs, 'pred_class': (probs > 0.5).astype(int)})
df_out.to_csv('results/predictions.csv', index=False)
print(df_out.head(10))
print(f"High expression predicted: {df_out['pred_class'].sum()}/{len(df_out)}")