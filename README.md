# DATA-BREACHES-2.0

## Overview

This repository provides a comprehensive framework for analyzing, modeling, and predicting data breaches using a variety of machine learning and deep learning techniques. The project is organized into several modules, each focusing on a different modeling approach, including Random Forests, Graph Attention Networks (GAT), Graph Convolutional Networks (GCN), and BERT-based models for feature extraction and embeddings.

---

## Project Structure

```
Data breaches 2.0/
├── bert_model/         # Pretrained BERT model and configs for embeddings
├── gat/                # Graph Attention Network (GAT) implementation
├── gcn/                # Graph Convolutional Network (GCN) implementation
├── rf/                 # Random Forest and related models
```

### bert_model/
- Contains a pretrained BERT model and configuration files for generating embeddings from textual data.
- Useful for feature extraction and transfer learning in downstream tasks.
- Includes:
  - Model weights (`model.safetensors`)
  - Tokenizer and config files
  - Pooling and normalization modules

### gat/
- Implements a Graph Attention Network for analyzing relationships in data breach records.
- Main files:
  - `gat.ipynb`: Jupyter notebook for GAT experiments
  - `Data_Breaches_K.csv`: Dataset used for graph-based modeling

### gcn/
- Implements a Graph Convolutional Network for similar graph-based analysis.
- Main files:
  - `gcn.ipynb`: Jupyter notebook for GCN experiments
  - `Data_Breaches_K.csv`: Dataset used for graph-based modeling

### rf/
- Contains Random Forest models and related preprocessing tools.
- Main files:
  - `Randomforest.ipynb`: Jupyter notebook for Random Forest experiments
  - Pretrained models and encoders (`rf_model.pkl`, `company_risk_model.pkl`, `le_sector.pkl`, `le_severity.pkl`, `scaler_records.pkl`, `scaler_year.pkl`, `sector_encoder.pkl`)
  - Copy of the BERT model for feature extraction
  - `Data_Breaches_K.csv`: Dataset for tabular modeling

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "Data breaches 2.0"
   ```
2. **Install dependencies:**
   - Make sure you have Python 3.7+ and Jupyter Notebook installed.
   - Install required Python packages (see individual notebooks for requirements, e.g., `transformers`, `scikit-learn`, `networkx`, `torch`, etc.).

3. **Run Notebooks:**
   - Open the desired notebook (e.g., `gat.ipynb`, `gcn.ipynb`, `Randomforest.ipynb`) in Jupyter and follow the instructions.

---

## Datasets
- `Data_Breaches_K.csv`: Main dataset used across all models. Contains records of data breaches with features suitable for both tabular and graph-based modeling.

---

## Pretrained Models & Encoders
- Pretrained models and encoders are provided in the `rf/` directory for immediate inference or further fine-tuning.
- The `bert_model/` directory contains all necessary files for BERT-based feature extraction.

---

## Usage Examples
- **Random Forest:**
  - Run `rf/Randomforest.ipynb` to train or evaluate the Random Forest model.
- **GAT/GCN:**
  - Run `gat/gat.ipynb` or `gcn/gcn.ipynb` for graph-based experiments.
- **BERT Embeddings:**
  - Use scripts or notebooks to generate embeddings from textual data using the files in `bert_model/`.

---

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License
Specify your license here (e.g., MIT, Apache 2.0, etc.).

---

## Acknowledgements
- Built with open-source libraries such as PyTorch, scikit-learn, transformers, and networkx.
- Inspired by research in data breach analysis and graph neural networks.
