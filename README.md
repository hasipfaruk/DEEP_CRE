# DeepCRE Model - Setup and Usage Guide

## 📋 What You Need

### 1. **Install Dependencies**

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install tensorflow==2.12.0 numpy pandas biopython shap scikit-learn matplotlib
```

### 2. **Prepare Your Data Files**

Create a `data/` directory in the `DeepCRE` folder and place these files:

#### Required Files:

1. **`data/genome.fasta`**
   - Genome sequence file in FASTA format
   - Each chromosome/scaffold as a separate sequence entry
   - Example:
     ```
     >chr1
     ATGCGATCGATCG...
     >chr2
     GCTAGCTAGCTA...
     ```

2. **`data/genes.gtf`** (or `data/genes.gff3`)
   - Gene annotation file in GTF or GFF3 format
   - Must contain gene features with `gene_id` in attributes
   - Example GTF format:
     ```
     chr1    source    gene    1000    5000    .    +    .    gene_id "gene001";
     ```

3. **`data/tpm_counts.csv`**
   - Expression data in CSV format
   - **Requirements:**
     - Gene IDs as **index** (first column, no header)
     - Must contain column named **`logMaxTPM`** (log-transformed TPM values)
   - Example:
     ```csv
     gene_id,logMaxTPM
     gene001,5.2
     gene002,3.1
     gene003,7.8
     ```

### 3. **Directory Structure**

Your project should look like this:

```
DeepCRE/
├── data/
│   ├── genome.fasta
│   ├── genes.gtf (or genes.gff3)
│   └── tpm_counts.csv
├── models/          (created automatically)
├── results/         (created automatically)
├── train_model.py
├── predict.py
├── interpret_model.py
├── utils.py
├── requirements.txt
└── README.md
```

## 🚀 How to Use

### Step 1: Train the Model

```bash
cd DeepCRE
python train_model.py
```

**What it does:**
- Extracts regulatory sequences (promoter, 5'UTR, 3'UTR, terminator) for each gene
- Creates binary labels (high/low expression) based on TPM quartiles
- Trains CNN model with chromosomal cross-validation
- Saves trained models in `models/` directory
- Saves training results in `results/training_results.csv`

**Output:**
- `models/cotton_fold0.h5`, `models/cotton_fold1.h5`, ... (one per fold)
- `models/best_cotton_model.h5` (best performing model)
- `results/training_results.csv` (performance metrics)

### Step 2: Make Predictions

```bash
python predict.py
```

**What it does:**
- Loads the best trained model
- Predicts expression levels for all genes
- Saves predictions to `results/predictions.csv`

**Output:**
- `results/predictions.csv` with columns:
  - `gene_id`: Gene identifier
  - `pred_high_prob`: Probability of high expression (0-1)
  - `pred_class`: Binary prediction (0=low, 1=high)

### Step 3: Interpret the Model (Optional)

```bash
python interpret_model.py
```

**What it does:**
- Uses SHAP to identify important sequence regions
- Generates visualization plots

**Output:**
- `results/shap_summary.png` - Feature importance summary
- `results/shap_force.png` - Sequence-level interpretation

## 📊 Data Format Details

### TPM Counts CSV Format

The `tpm_counts.csv` file must have:
- **Index column**: Gene IDs (matching gene_id in GTF file)
- **Required column**: `logMaxTPM` (log-transformed maximum TPM across conditions)

Example:
```csv
,logMaxTPM
Gorai.001G000100,5.234
Gorai.001G000200,3.456
Gorai.001G000300,7.890
```

### GTF/GFF3 Format

The annotation file should contain gene features. The code automatically:
- Extracts gene coordinates (start, end, strand)
- Parses `gene_id` from attributes
- Handles both GTF and GFF3 formats

## ⚠️ Troubleshooting

### Common Issues:

1. **FileNotFoundError: data/tpm_counts.csv**
   - Make sure `data/` directory exists and contains all required files

2. **ValueError: tpm_df must contain 'logMaxTPM' column**
   - Check your CSV has a column named exactly `logMaxTPM`
   - Ensure it's log-transformed TPM values

3. **KeyError when accessing gene_id**
   - Ensure gene IDs in GTF match gene IDs in TPM CSV (as index)

4. **No valid sequences extracted**
   - Check chromosome names match between FASTA and GTF
   - Verify gene coordinates are within genome bounds
   - Ensure gene IDs are present in both GTF and TPM files

5. **Out of memory errors**
   - Reduce batch size in training (modify `batch_size=32` to smaller value)
   - Use fewer samples for SHAP interpretation

## 📝 Notes

- The model extracts ~3000bp sequences per gene (1kb promoter + 0.5kb 5'UTR + 0.5kb 3'UTR + 1kb terminator)
- Only genes in Q25 (low) or Q75 (high) expression quartiles are used for training
- Chromosomal cross-validation ensures no chromosome overlap between train/validation sets
- Training includes a shuffled control experiment to validate sequence-specific learning

## 🔧 Customization

You can modify:
- Sequence lengths in `utils.py` (currently 3000bp)
- CNN architecture in `build_cnn()` function
- Training parameters (epochs, batch_size) in `train_model.py`
- Expression thresholds (currently Q25/Q75) in `utils.py`

