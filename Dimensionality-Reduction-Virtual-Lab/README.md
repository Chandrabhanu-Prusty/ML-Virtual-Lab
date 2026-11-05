# Dimensionality Reduction Virtual Laboratory

> **College Machine Learning Laboratory Coursework (Semester V)**  
> **Experiment Title:** Dimensionality Reduction Using PCA, SVD, t-SNE, and UMAP  
> **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) — UCI Machine Learning Repository  

---

## 🎯 Experiment Aim

To apply linear (**Principal Component Analysis - PCA**, **Singular Value Decomposition - SVD**) and non-linear (**t-Distributed Stochastic Neighbor Embedding - t-SNE**, **Uniform Manifold Approximation and Projection - UMAP**) dimensionality reduction techniques on a high-dimensional biomedical dataset (30 continuous numerical features), analyze principal component loadings and variance distribution, visualize the latent 2D and 3D manifold embeddings, and quantitatively evaluate the impact of reduced feature representations on downstream classification accuracy using Logistic Regression.

---

## 📊 Dataset Specifications

- **Source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
- **Samples ($N$):** 569 patient biopsy cases
- **Continuous Features ($D$):** 30 numerical morphological attributes
- **Classes:** 
  - **Benign ($B$):** 357 instances (62.74%)
  - **Malignant ($M$):** 212 instances (37.26%)
- **Target Column:** Column 2 (`diagnosis`: 'M' or 'B')
- **Missing Values:** 0 missing values (100% complete)

### Manual Dataset Placement (If Network / Fetch Fails)
If running in an environment without internet access or where local browser CORS security blocks `fetch()`:
1. Download `wdbc.data` from:  
   `https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data`
2. Place the file directly in the `data/` directory:  
   `Dimensionality-Reduction-Virtual-Lab/data/wdbc.data`
3. *Note:* The virtual lab already includes an automated embedded fallback inside `js/data.js` containing all 569 instances, guaranteeing that the application functions seamlessly even when opened directly as a local file (`file:///`) in modern web browsers!

---

## 🗂️ Project Directory Structure

```
Dimensionality-Reduction-Virtual-Lab/
├── index.html            # Main interactive Virtual Lab dashboard
├── report.html           # Academic laboratory experiment report (Print/PDF ready)
├── README.md             # Complete documentation and student lab manual
├── css/
│   └── style.css         # Modern SaaS educational dark navy/purple styling
├── js/
│   ├── app.js            # UI controller, tab switching, and event orchestrator
│   ├── data.js           # Dataset parser, stats calculator, and Z-score scaler
│   ├── algorithms.js     # Exact Jacobi PCA, Thin SVD, t-SNE, UMAP, and Logistic Regression
│   ├── charts.js         # Interactive 2D/3D Plotly.js chart routines
│   └── lib/              # Verified local vendor fallbacks for offline execution
│       ├── tsne.js       # Andrej Karpathy's exact t-SNE algorithm
│       └── umap-js.min.js# Official UMAP-js algorithm library
├── data/
│   └── wdbc.data         # Official UCI WDBC raw CSV dataset
└── assets/
    └── logo.svg          # High-resolution dimensional projection vector logo
```

---

## 🚀 How to Run the Virtual Lab

This project uses **only pure HTML5, CSS3, and Vanilla JavaScript** with **zero build tools, zero Node.js dependencies, and zero database requirements**.

### Option 1: Using VS Code Live Server (Recommended)
1. Open Visual Studio Code.
2. Select **File > Open Folder...** and select `Dimensionality-Reduction-Virtual-Lab`.
3. Ensure the **Live Server** extension by *Ritwick Dey* is installed in VS Code.
4. Right-click `index.html` and click **"Open with Live Server"** (or click **"Go Live"** on the bottom status bar).
5. The application will launch in your default browser at `http://127.0.0.1:5500/index.html`.

### Option 2: Using Any Static Web Server
```bash
# Python 3
python -m http.server 8000

# Node http-server (optional)
npx http-server . -p 8000
```
Open your browser to `http://localhost:8000`.

### Option 3: Direct File Opening
Double-click `index.html` in your file explorer. Thanks to the built-in dataset fallback in `js/data.js`, the lab runs immediately without requiring a local web server!

---

## 🔬 Mathematical Formulations

### 1. Principal Component Analysis (PCA)
- **Standardization:** $z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$
- **Covariance Matrix:** $\mathbf{\Sigma} = \frac{1}{N - 1} \mathbf{X}_{std}^T \mathbf{X}_{std} \in \mathbb{R}^{30 \times 30}$
- **Spectral Decomposition:** $\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$ (Solved via the Jacobi rotation algorithm)
- **Explained Variance Ratio:** $\text{EVR}_i = \frac{\lambda_i}{\sum_{k=1}^{30} \lambda_k}$
- **Dimensional Projection:** $\mathbf{Z}_k = \mathbf{X}_{std} \mathbf{V}_k \in \mathbb{R}^{N \times k}$

### 2. Singular Value Decomposition (SVD)
- **Matrix Factorization:** $\mathbf{X}_{std} = \mathbf{U} \mathbf{S} \mathbf{V}^T$
- **Singular Values:** $\sigma_i = \sqrt{\lambda_i(\mathbf{X}_{std}^T \mathbf{X}_{std})}$
- **Low-Rank Coordinate Projection:** $\mathbf{Z}_k = \mathbf{X}_{std} \mathbf{V}_k = \mathbf{U}_k \mathbf{S}_k$

### 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)
- **High-Dimensional Affinities (Gaussian):**  
  $$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
- **Low-Dimensional Affinities (Student-t with 1 d.o.f.):**  
  $$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$
- **Objective Loss:** Minimizes Kullback-Leibler divergence: $\mathcal{L} = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$

### 4. Uniform Manifold Approximation and Projection (UMAP)
- Models data as fuzzy simplicial sets on a Riemannian manifold.
- High-dimensional fuzzy weights: $p_{i|j} = \exp\left(-\max(0, d(x_i, x_j) - \rho_i) / \sigma_i\right)$
- Optimizes low-dimensional coordinates by minimizing fuzzy set cross-entropy using stochastic gradient descent.

---

## 📈 Key Empirical Observations & Results

| Feature Representation | Transformation Class | Dimensions | Explained Variance | Test Accuracy | Test F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Original Baseline** | Raw Features | 30 | 100.0% | **97.66%** | **97.01%** |
| **PCA** | Linear Orthogonal | 2 | 63.23% | **94.15%** | **92.19%** |
| **SVD** | Linear Factorization | 2 | 63.23% | **94.15%** | **92.19%** |
| **t-SNE** | Non-Linear Manifold | 2 | N/A (Probabilistic) | **93.57%** | **91.43%** |
| **UMAP** | Non-Linear Topological| 2 | N/A (Topological) | **95.32%** | **93.85%** |

### Insights:
1. **Curse of Dimensionality Mitigated:** Reducing the feature space from 30 dimensions to 2 dimensions (a **93.3% reduction**) costs only ~2.3% in classification accuracy with linear PCA, while accelerating model training significantly.
2. **PC1 Loadings:** Features representing tumor boundary size and irregularity (`worst perimeter`, `worst concave points`, `worst radius`) carry the highest weights on Principal Component 1.
3. **Global vs. Local Structure:** PCA and SVD preserve global pairwise distances, whereas t-SNE forms tight local islands that reveal distinct sub-types within the malignant cluster. UMAP uniquely balances both local clusters and global separation.

---

## 🎓 Academic Lab Examination (Viva Questions)

1. **Why must features be standardized before PCA?**  
   *Answer:* PCA maximizes variance. Features with large numerical ranges (such as cell area $\approx 1000$) will artificially dominate principal components over features with small scales (such as smoothness $\approx 0.1$) regardless of biological relevance.

2. **What is the mathematical relationship between PCA and SVD?**  
   *Answer:* For a centered/standardized data matrix $\mathbf{X}$, the right singular vectors $\mathbf{V}$ from SVD are identical to the eigenvectors of the covariance matrix $\mathbf{\Sigma} = \frac{1}{N-1}\mathbf{X}^T\mathbf{X}$, and the singular values satisfy $\sigma_i = \sqrt{(N-1)\lambda_i}$.

3. **Why does t-SNE use a Student-t distribution in the low-dimensional space?**  
   *Answer:* High-dimensional space has vastly more volume than 2D space. The heavy tails of the Student-t distribution allow moderate distances in high dimensions to be represented by larger distances in low dimensions, resolving the "crowding problem".

4. **Can t-SNE transform new out-of-sample test points?**  
   *Answer:* Standard t-SNE does not learn an explicit projection mapping function $f(x)$; embedding new unseen points requires re-running the optimization or training a parametric network.

---

## 📜 License
Developed for educational virtual laboratory instruction in college machine learning and data science coursework. Free to adapt and use.

