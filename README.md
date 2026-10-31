# Citation Prediction

Predicts whether one scientific paper cites another, as a **link prediction** problem on a citation graph. For each pair of papers the model combines **graph features**, **author overlap** and the **semantic similarity of their abstracts**, and an **XGBoost** classifier outputs the probability that a citation exists.

Developed for the **Natural Language Processing** course at the Department of Computer Science & Engineering, University of Ioannina (2025), as part of the course's Kaggle challenge.

## Methodology

1. **Text processing:** abstracts are lowercased and stripped of punctuation, then encoded with the **`all-mpnet-base-v2`** Sentence-Transformers model. The embeddings are L2-normalized, so their dot product is the cosine similarity.
2. **Training pairs:** every edge of the citation graph is a positive example. An equal number of random pairs of papers that are not connected are used as negative examples.
3. **Features** for each pair of papers *(u, v)*:
   - Number of common neighbors
   - Jaccard similarity of their neighborhoods
   - Preferential attachment (product of degrees)
   - Adamic-Adar index
   - Number of shared authors
   - Cosine similarity of the abstract embeddings
4. **Model:** an **XGBoost** classifier (1000 trees, depth 6, early stopping on the validation set), trained on 80% of the pairs and evaluated on the remaining 20%.
5. **Prediction:** the same features are computed for the pairs in `test.txt`, and the predicted probabilities are written to `submission.csv`.

### Design choices
- **Text and graph features together:** the cosine similarity of the abstract embeddings is the most informative feature, and it complements the graph features.
- **Embedding model:** `all-mpnet-base-v2` (768 dimensions) produces high-quality sentence embeddings, but it is slow: embedding all abstracts takes about 1 hour on a GTX 1650 Ti GPU and about 12 hours on a CPU, which is why the computed features are included in the repository.
- **Balanced negative sampling:** one random unconnected pair for every citation (1:1).
- **XGBoost tuning:** a small learning rate (0.05) with up to 1000 trees, maximum depth 6, 80% row and column subsampling, L1/L2 regularization, and early stopping after 50 rounds without improvement.

## Results
| Set | Log loss |
|---|---|
| Kaggle private leaderboard | **0.1149** |
| Kaggle public leaderboard | 0.1187 |
| Validation set (20% of the training pairs) | ≈ 0.121 |

The validation score can differ slightly between XGBoost versions.

## Data
The repository includes the **precomputed features** (`train_features.npz`, `val_features.npz`, `test_features.npz`), since computing the abstract embeddings takes a long time. When these files are present, the notebook loads them directly and does not need the original data.

To compute the features from scratch, delete the `.npz` files and place the original data files (not included in the repository) in the same folder:

| File | Format | Content |
|---|---|---|
| `abstracts.txt` | `paper_id\|--\|abstract` | Abstract of each paper |
| `authors.txt` | `paper_id\|--\|author1,author2,...` | Authors of each paper |
| `edgelist.txt` | `paper_id,paper_id` | Citations between papers |
| `test.txt` | `paper_id,paper_id` | Pairs to predict |

## Requirements
- Python 3.10+
- The packages in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
A GPU is optional; if one is available, it is used for the embeddings and for XGBoost.

## Usage
Open `citation_prediction.ipynb` in Jupyter and run all cells. With the included `.npz` files, the notebook trains the model and makes the predictions directly. The predictions are saved to `submission.csv` (columns `ID`, `Label`).

## Authors

**Athanasios Tonios**
- [GitHub](https://github.com/athanasios-tonios)

**Panagiotis Trypos**
- [GitHub](https://github.com/ptrypos)
- [Website / Portfolio](https://ptrypos.me)
- [LinkedIn](https://www.linkedin.com/in/%CF%80%CE%B1%CE%BD%CE%B1%CE%B3%CE%B9%CF%8E%CF%84%CE%B7%CF%82-%CF%84%CF%81%CF%8D%CF%80%CE%BF%CF%82-983abb242)
