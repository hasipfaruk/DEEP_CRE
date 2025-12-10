import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from io import StringIO  # For GFF3 to GTF conversion if needed

def gff3_to_gtf(gff3_path, output_path):
    """Convert GFF3 to GTF format (basic)."""
    with open(gff3_path, 'r') as f, open(output_path, 'w') as out:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            chr, source, feature, start, end, score, strand, frame, attrs = fields
            attrs_dict = dict(attr.split('=') for attr in attrs.split(';') if '=' in attr)
            gene_id = attrs_dict.get('ID', attrs_dict.get('Parent', 'unknown'))
            out.write(f"{chr}\t{source}\t{feature}\t{int(start)}\t{int(end)}\t{score}\t{strand}\t{frame}\tgene_id \"{gene_id}\";\n")

def one_hot_encode(seq):
    """One-hot encode DNA sequence (A/C/G/T/N -> [1,0,0,0] etc.)."""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0.25]*4}
    seq = str(seq).upper().replace(' ', '')  # Clean
    return np.array([mapping.get(base, [0]*4) for base in seq])

class SequenceLoader:
    def __init__(self, genome_path, gtf_path):
        self.genome = {rec.id: rec.seq for rec in SeqIO.parse(genome_path, "fasta")}
        # Load GTF (or convert GFF3)
        if gtf_path.endswith('.gff3'):
            gtf_temp = gtf_path.replace('.gff3', '_temp.gtf')
            gff3_to_gtf(gtf_path, gtf_temp)
            gtf_path = gtf_temp
        self.gtf = pd.read_csv(gtf_path, sep='\t', comment='#', header=None,
                               names=['chr', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])
        self.gtf['gene_id'] = self.gtf['attribute'].str.extract(r'gene_id "([^"]+)"')

    def extract_sequences(self, tpm_df):
        """Extract flanking sequences and binary labels (low=0, high=1)."""
        genes = self.gtf[self.gtf['feature'] == 'gene'].dropna(subset=['gene_id'])
        genes = genes[genes['gene_id'].isin(tpm_df.index)]

        sequences, labels, ids = [], [], []
        q25, q75 = tpm_df['logMaxTPM'].quantile([0.25, 0.75])

        for _, row in genes.iterrows():
            chr_name = row['chr']
            if chr_name not in self.genome:
                continue  # Skip missing chr
            seq = self.genome[chr_name]
            start, end, strand = int(row['start']), int(row['end']), row['strand']

            # TSS/TTS based on strand
            tss = start if strand == '+' else end
            tts = end if strand == '+' else start

            # Extract regions (paper: 1kb up/down + 0.5kb UTRs)
            if strand == '+':
                promoter = seq[max(0, tss-1001):tss-1]
                utr5 = seq[tss-1:tss+499]
                utr3 = seq[tts-500:tts]
                terminator = seq[tts:tts+1000]
            else:
                rev_seq = seq.reverse_complement()
                promoter = rev_seq[end:end+1000]  # Adjust for RC
                utr5 = rev_seq[end-499:end]
                utr3 = rev_seq[start:start+500]
                terminator = rev_seq[start-1001:start-1]

            full_seq = (str(promoter) + str(utr5) + 'N'*20 + str(utr3) + str(terminator))
            encoded = one_hot_encode(full_seq)
            if len(encoded) < 3000:  # Pad if short
                encoded = np.pad(encoded, ((0, 3000 - len(encoded)), (0,0)), 'constant')

            logtpm = tpm_df.loc[row['gene_id'], 'logMaxTPM']
            label = 0 if logtpm <= q25 else (1 if logtpm > q75 else -1)
            if label != -1:
                sequences.append(encoded[:3000])  # Fixed length
                labels.append(label)
                ids.append(row['gene_id'])

        return np.array(sequences), np.array(labels), ids

def build_cnn(input_shape=(3000, 4)):
    """CNN architecture from DeepCRE (3 conv blocks)."""
    model = Sequential([
        # Block 1
        Conv1D(64, 5, activation='relu', input_shape=input_shape),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2), Dropout(0.25),
        # Block 2
        Conv1D(128, 5, activation='relu'),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(2), Dropout(0.25),
        # Block 3
        Conv1D(256, 5, activation='relu'),
        Conv1D(256, 5, activation='relu'),
        MaxPooling1D(2), Dropout(0.25),
        # Dense
        Flatten(),
        Dense(512, activation='relu'), Dropout(0.5),
        Dense(256, activation='relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')])
    return model