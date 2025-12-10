"""
Helper script to create the required directory structure for DeepCRE.
Run this script to set up the data directory structure.
"""

import os

def setup_directories():
    """Create necessary directories for DeepCRE model."""
    directories = [
        'data',
        'models',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")
    
    print("\n📁 Directory structure created!")
    print("\n📋 Next steps:")
    print("1. Place your genome.fasta file in: data/genome.fasta")
    print("2. Place your genes.gtf (or genes.gff3) file in: data/genes.gtf")
    print("3. Place your tpm_counts.csv file in: data/tpm_counts.csv")
    print("\nThe tpm_counts.csv should have:")
    print("  - Gene IDs as index (first column)")
    print("  - A column named 'logMaxTPM' with log-transformed TPM values")
    print("\nThen run: python train_model.py")

if __name__ == "__main__":
    setup_directories()

