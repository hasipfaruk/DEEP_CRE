import os
import tensorflow as tf
from utils import SequenceLoader, build_cnn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load data
tpm_df = pd.read_csv('data/tpm_counts.csv', index_col=0)
loader = SequenceLoader('data/genome.fasta', 'data/genes.gtf')
X, y, ids = loader.extract_sequences(tpm_df)

# Chromosomal CV (adapt chromosomes, e.g., A01-D13 for cotton)
# Map each gene to its chromosome
gene_to_chr = {}
for gene_id in ids:
    gene_rows = loader.gtf[loader.gtf['gene_id'] == gene_id]
    if len(gene_rows) > 0:
        gene_to_chr[gene_id] = gene_rows.iloc[0]['chr']
    else:
        gene_to_chr[gene_id] = 'unknown'

chr_list = [gene_to_chr.get(gid, 'unknown') for gid in ids]
unique_chrs = sorted(set(chr_list))

# Split by chromosomes, not by samples
kf = KFold(n_splits=min(5, len(unique_chrs)), shuffle=True, random_state=42)
results = []

for fold, (train_chr_idx, val_chr_idx) in enumerate(kf.split(unique_chrs)):
    train_chrs = set([unique_chrs[i] for i in train_chr_idx])
    val_chrs = set([unique_chrs[i] for i in val_chr_idx])
    
    # Create masks based on chromosome membership
    train_mask = np.array([chr_list[i] in train_chrs for i in range(len(X))])
    val_mask = np.array([chr_list[i] in val_chrs for i in range(len(X))])
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    # Normal model
    model = build_cnn()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                                   tf.keras.callbacks.ModelCheckpoint(f'models/cotton_fold{fold}.h5', save_best_only=True)])
    val_loss, val_acc, val_auroc = model.evaluate(X_val, y_val, verbose=0)
    results.append({'fold': fold, 'type': 'normal', 'acc': val_acc, 'auroc': val_auroc})

    # Shuffled control
    X_train_shuf = X_train.copy()
    for i in range(len(X_train_shuf)):
        np.random.shuffle(X_train_shuf[i])  # Shuffle sequences
    model_shuf = build_cnn()
    model_shuf.fit(X_train_shuf, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)
    val_loss_shuf, val_acc_shuf, val_auroc_shuf = model_shuf.evaluate(X_val, y_val, verbose=0)
    results.append({'fold': fold, 'type': 'shuffled', 'acc': val_acc_shuf, 'auroc': val_auroc_shuf})

# Save results
pd.DataFrame(results).to_csv('results/training_results.csv', index=False)
print(pd.DataFrame(results).groupby('type')[['acc', 'auroc']].mean())

# Save best model (select fold with best AUROC)
normal_results = [r for r in results if r['type'] == 'normal']
if normal_results:
    best_fold = max(normal_results, key=lambda x: x['auroc'])['fold']
    best_model = tf.keras.models.load_model(f'models/cotton_fold{best_fold}.h5')
    best_model.save('models/best_cotton_model.h5')
    print(f"Best model saved from fold {best_fold} with AUROC: {max(normal_results, key=lambda x: x['auroc'])['auroc']:.4f}")
else:
    print("Warning: No normal model results found. Using fold 0 as default.")
    best_model = tf.keras.models.load_model('models/cotton_fold0.h5')
    best_model.save('models/best_cotton_model.h5')