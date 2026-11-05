/**
 * Dimensionality Reduction Virtual Lab
 * Algorithms Module: PCA, SVD, t-SNE, UMAP & Classification
 */

// ==========================================
// 1. LINEAR ALGEBRA & MATHEMATICAL ROUTINES
// ==========================================

class MathLib {
  /**
   * Computes sample covariance matrix of standardized data X_std (N x M)
   * Cov = (1 / (N - 1)) * (X^T * X)
   */
  static covarianceMatrix(X) {
    const N = X.length;
    const M = X[0].length;
    const cov = Array.from({ length: M }, () => new Float64Array(M));

    for (let i = 0; i < M; i++) {
      for (let j = i; j < M; j++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
          sum += X[k][i] * X[k][j];
        }
        const val = sum / (N - 1);
        cov[i][j] = val;
        cov[j][i] = val;
      }
    }
    return cov;
  }

  /**
   * Computes Gram matrix X^T * X for SVD
   */
  static gramMatrix(X) {
    const N = X.length;
    const M = X[0].length;
    const G = Array.from({ length: M }, () => new Float64Array(M));

    for (let i = 0; i < M; i++) {
      for (let j = i; j < M; j++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
          sum += X[k][i] * X[k][j];
        }
        G[i][j] = sum;
        G[j][i] = sum;
      }
    }
    return G;
  }

  /**
   * Jacobi Eigenvalue Algorithm for real symmetric matrices
   * Computes all eigenvalues and eigenvectors with high precision
   */
  static jacobiEigen(A, maxIter = 150, eps = 1e-12) {
    const n = A.length;
    // Initialize V as identity matrix
    const V = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      row[i] = 1.0;
      return row;
    });

    // Copy A into D
    const D = A.map(row => new Float64Array(row));

    for (let iter = 0; iter < maxIter; iter++) {
      // Find largest off-diagonal element
      let maxOffDiag = 0;
      let p = 0, q = 1;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const absVal = Math.abs(D[i][j]);
          if (absVal > maxOffDiag) {
            maxOffDiag = absVal;
            p = i;
            q = j;
          }
        }
      }

      if (maxOffDiag < eps) break;

      const app = D[p][p];
      const aqq = D[q][q];
      const apq = D[p][q];

      // Jacobi rotation angle
      const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
      const c = Math.cos(theta);
      const s = Math.sin(theta);

      // Update eigenvectors V
      for (let i = 0; i < n; i++) {
        const vip = V[i][p];
        const viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
      }

      // Update matrix D
      for (let i = 0; i < n; i++) {
        if (i !== p && i !== q) {
          const dip = D[i][p];
          const diq = D[i][q];
          const newDip = c * dip - s * diq;
          const newDiq = s * dip + c * diq;
          D[i][p] = D[p][i] = newDip;
          D[i][q] = D[q][i] = newDiq;
        }
      }

      const dpp = c * c * app - 2 * s * c * apq + s * s * aqq;
      const dqq = s * s * app + 2 * s * c * apq + c * c * aqq;
      D[p][p] = dpp;
      D[q][q] = dqq;
      D[p][q] = D[q][p] = 0;
    }

    // Extract eigenvalues and eigenvectors
    const eigen = [];
    for (let i = 0; i < n; i++) {
      const vec = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        vec[j] = V[j][i];
      }
      eigen.push({
        value: Math.max(0, D[i][i]), // Numerical stabilization
        vector: Array.from(vec)
      });
    }

    // Sort descending by eigenvalue
    eigen.sort((a, b) => b.value - a.value);
    return eigen;
  }

  /**
   * Projects matrix X (N x M) onto top k components V_k (M x k)
   */
  static project(X, components) {
    const N = X.length;
    const k = components.length;
    const result = Array.from({ length: N }, () => new Float64Array(k));

    for (let i = 0; i < N; i++) {
      for (let c = 0; c < k; c++) {
        let dot = 0;
        const vec = components[c];
        const row = X[i];
        for (let j = 0; j < row.length; j++) {
          dot += row[j] * vec[j];
        }
        result[i][c] = dot;
      }
    }
    return result.map(row => Array.from(row));
  }
}

// ==========================================
// 2. PRINCIPAL COMPONENT ANALYSIS (PCA)
// ==========================================

class PCARunner {
  static run(X_std, nComponents = 2) {
    const startTime = performance.now();
    const N = X_std.length;
    const M = X_std[0].length;

    // 1. Covariance matrix (30x30)
    const cov = MathLib.covarianceMatrix(X_std);

    // 2. Exact Jacobi Eigen-decomposition
    const eigen = MathLib.jacobiEigen(cov);

    // 3. Total variance & explained variance ratios
    const totalVariance = eigen.reduce((sum, e) => sum + e.value, 0);
    const explainedVarianceRatio = eigen.map(e => e.value / (totalVariance || 1e-10));

    let cum = 0;
    const cumulativeVariance = explainedVarianceRatio.map(ratio => {
      cum += ratio;
      return cum;
    });

    // 4. Select top k components
    const topComponents = eigen.slice(0, nComponents).map(e => e.vector);

    // 5. Project data to k dimensions
    const projections = MathLib.project(X_std, topComponents);

    // 6. Feature Loadings & Importance Analysis
    // Loading L_ij = eigenvector_j[i] * sqrt(eigenvalue_j)
    const loadings = [];
    for (let i = 0; i < M; i++) {
      const featureLoadings = [];
      for (let c = 0; c < nComponents; c++) {
        featureLoadings.push(topComponents[c][i] * Math.sqrt(eigen[c].value));
      }
      loadings.push(featureLoadings);
    }

    // Absolute contribution to PC1 and PC2
    const featureContributions = [];
    for (let i = 0; i < M; i++) {
      const pc1Weight = Math.abs(topComponents[0][i]);
      const pc2Weight = nComponents > 1 ? Math.abs(topComponents[1][i]) : 0;
      const combined = (pc1Weight * explainedVarianceRatio[0] + pc2Weight * (explainedVarianceRatio[1] || 0));
      featureContributions.push({
        featureIndex: i,
        name: FEATURE_NAMES[i],
        label: FEATURE_LABELS[i],
        pc1Weight: parseFloat(topComponents[0][i].toFixed(4)),
        pc2Weight: nComponents > 1 ? parseFloat(topComponents[1][i].toFixed(4)) : 0,
        pc1Loading: parseFloat(loadings[i][0].toFixed(4)),
        pc2Loading: nComponents > 1 ? parseFloat(loadings[i][1].toFixed(4)) : 0,
        score: combined
      });
    }

    featureContributions.sort((a, b) => b.score - a.score);

    const endTime = performance.now();

    return {
      algorithm: 'PCA',
      projections,
      nComponents,
      eigenvalues: eigen.map(e => parseFloat(e.value.toFixed(6))),
      explainedVarianceRatio: explainedVarianceRatio.map(r => parseFloat((r * 100).toFixed(3))),
      cumulativeVariance: cumulativeVariance.map(c => parseFloat((c * 100).toFixed(3))),
      totalExplainedVariance: parseFloat((cumulativeVariance[nComponents - 1] * 100).toFixed(2)),
      featureContributions,
      executionTimeMs: Math.round(endTime - startTime),
      components: topComponents
    };
  }
}

// ==========================================
// 3. SINGULAR VALUE DECOMPOSITION (SVD)
// ==========================================

class SVDRunner {
  static run(X_std, nComponents = 2) {
    const startTime = performance.now();
    const N = X_std.length;
    const M = X_std[0].length;

    // Thin SVD via Gram Matrix G = X^T * X
    // G = V * S^2 * V^T where S is singular values diagonal
    const G = MathLib.gramMatrix(X_std);
    const eigen = MathLib.jacobiEigen(G);

    // Singular values = sqrt(eigenvalues of X^T * X)
    const singularValues = eigen.map(e => Math.sqrt(Math.max(0, e.value)));
    const totalEnergy = singularValues.reduce((sum, s) => sum + s * s, 0);

    const explainedVarianceRatio = singularValues.map(s => (s * s) / (totalEnergy || 1e-10));
    let cum = 0;
    const cumulativeVariance = explainedVarianceRatio.map(r => {
      cum += r;
      return cum;
    });

    // Right singular vectors V (columns of V)
    const rightSingularVectors = eigen.slice(0, nComponents).map(e => e.vector);

    // Coordinate projection: Z = X * V_k
    const projections = MathLib.project(X_std, rightSingularVectors);

    const endTime = performance.now();

    return {
      algorithm: 'SVD',
      projections,
      nComponents,
      singularValues: singularValues.map(s => parseFloat(s.toFixed(4))),
      explainedVarianceRatio: explainedVarianceRatio.map(r => parseFloat((r * 100).toFixed(3))),
      cumulativeVariance: cumulativeVariance.map(c => parseFloat((c * 100).toFixed(3))),
      totalExplainedVariance: parseFloat((cumulativeVariance[nComponents - 1] * 100).toFixed(2)),
      executionTimeMs: Math.round(endTime - startTime),
      rightSingularVectors
    };
  }
}

// ==========================================
// 4. t-SNE (Karpathy's Barnes-Hut / exact t-SNE)
// ==========================================

class TSNERunner {
  /**
   * Runs t-SNE with step callback option
   */
  static async run(X_data, options = {}, progressCallback = null) {
    const startTime = performance.now();
    const dim = options.dim || 2;
    const perplexity = options.perplexity || 30;
    const epsilon = options.epsilon || 10;
    const iterations = options.iterations || 200;

    // Ensure tsne library is initialized
    let TSNEClass = null;
    if (typeof tsnejs !== 'undefined' && tsnejs.tSNE) {
      TSNEClass = tsnejs.tSNE;
    } else if (typeof window.tSNE !== 'undefined') {
      TSNEClass = window.tSNE;
    } else {
      throw new Error('t-SNE library (tsnejs) not found. Please verify CDN or network connection.');
    }

    const tsne = new TSNEClass({
      dim: dim,
      perplexity: perplexity,
      epsilon: epsilon
    });

    tsne.initDataRaw(X_data);

    // Iterative stepping
    const stepBatch = Math.max(1, Math.floor(iterations / 20));
    for (let k = 0; k < iterations; k++) {
      tsne.step();
      if (progressCallback && k % stepBatch === 0) {
        const pct = Math.round(((k + 1) / iterations) * 100);
        progressCallback(pct, k + 1, iterations);
        // Yield momentarily to let UI update
        await new Promise(r => setTimeout(r, 0));
      }
    }

    const solution = tsne.getSolution();
    const endTime = performance.now();

    return {
      algorithm: 't-SNE',
      projections: solution,
      nComponents: dim,
      perplexity,
      epsilon,
      iterations,
      executionTimeMs: Math.round(endTime - startTime)
    };
  }
}

// ==========================================
// 5. UMAP (Uniform Manifold Approximation & Projection)
// ==========================================

class UMAPRunner {
  static async run(X_data, options = {}, progressCallback = null) {
    const startTime = performance.now();
    const nComponents = options.nComponents || 2;
    const nNeighbors = options.nNeighbors || 15;
    const minDist = options.minDist !== undefined ? options.minDist : 0.1;
    const spread = options.spread || 1.0;
    const nEpochs = options.nEpochs || 200;

    // Detect UMAP constructor from umap-js CDN/bundle
    let UMAPClass = null;
    if (typeof window.UMAP !== 'undefined') {
      if (typeof window.UMAP === 'function') {
        UMAPClass = window.UMAP;
      } else if (typeof window.UMAP.UMAP === 'function') {
        UMAPClass = window.UMAP.UMAP;
      }
    }

    if (!UMAPClass) {
      throw new Error('UMAP library (umap-js) not loaded. Please verify script tag or network connection.');
    }

    const umap = new UMAPClass({
      nComponents,
      nNeighbors,
      minDist,
      spread,
      nEpochs
    });

    if (progressCallback) progressCallback(30, 'Optimizing Riemannian manifold...');
    await new Promise(r => setTimeout(r, 10));

    // Fit embeddings
    const embedding = umap.fit(X_data);

    if (progressCallback) progressCallback(100, 'Embedding complete');
    const endTime = performance.now();

    return {
      algorithm: 'UMAP',
      projections: embedding,
      nComponents,
      nNeighbors,
      minDist,
      spread,
      nEpochs,
      executionTimeMs: Math.round(endTime - startTime)
    };
  }
}

// ==========================================
// 6. LOGISTIC REGRESSION CLASSIFIER & EVALUATION
// ==========================================

class ClassifierEvaluator {
  /**
   * Trains a Logistic Regression model using Gradient Descent
   */
  static trainLogisticRegression(X, y, epochs = 250, lr = 0.15) {
    const n = X.length;
    const m = X[0].length;
    const weights = new Float64Array(m);
    let bias = 0;

    for (let ep = 0; ep < epochs; ep++) {
      const dw = new Float64Array(m);
      let db = 0;

      for (let i = 0; i < n; i++) {
        let z = bias;
        const row = X[i];
        for (let j = 0; j < m; j++) {
          z += weights[j] * row[j];
        }
        // Sigmoid with numerical clamping
        const pred = 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));
        const err = pred - y[i];

        for (let j = 0; j < m; j++) {
          dw[j] += err * row[j];
        }
        db += err;
      }

      const step = lr / n;
      for (let j = 0; j < m; j++) {
        weights[j] -= step * dw[j];
      }
      bias -= step * db;
    }

    return { weights: Array.from(weights), bias };
  }

  /**
   * Evaluates classification metrics on test data
   */
  static evaluate(model, X_test, y_test) {
    const n = X_test.length;
    const m = X_test[0].length;
    let correct = 0;
    let tp = 0, fp = 0, fn = 0, tn = 0;

    for (let i = 0; i < n; i++) {
      let z = model.bias;
      const row = X_test[i];
      for (let j = 0; j < m; j++) {
        z += model.weights[j] * row[j];
      }
      const prob = 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));
      const pred = prob >= 0.5 ? 1 : 0;
      const actual = y_test[i];

      if (pred === actual) correct++;
      if (pred === 1 && actual === 1) tp++;
      if (pred === 1 && actual === 0) fp++;
      if (pred === 0 && actual === 1) fn++;
      if (pred === 0 && actual === 0) tn++;
    }

    const accuracy = (correct / n) * 100;
    const precision = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
    const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      accuracy: parseFloat(accuracy.toFixed(2)),
      precision: parseFloat(precision.toFixed(2)),
      recall: parseFloat(recall.toFixed(2)),
      f1: parseFloat(f1.toFixed(2)),
      tp, fp, fn, tn,
      totalSamples: n
    };
  }

  /**
   * Evaluates classification accuracy for a given feature set
   */
  static evaluateFeatureSet(X_features, y_all, split) {
    const t0 = performance.now();
    const X_train = split.trainIndices.map(i => X_features[i]);
    const y_train = split.trainIndices.map(i => y_all[i]);
    const X_test = split.testIndices.map(i => X_features[i]);
    const y_test = split.testIndices.map(i => y_all[i]);

    const model = this.trainLogisticRegression(X_train, y_train, 250, 0.2);
    const metrics = this.evaluate(model, X_test, y_test);
    metrics.evalTimeMs = Math.round(performance.now() - t0);
    return metrics;
  }
}

// ==========================================
// 7. COMPREHENSIVE EXPERIMENT WORKFLOW
// ==========================================

class ExperimentSuite {
  /**
   * Runs the complete benchmark suite on the active dataset
   */
  static async runFullExperiment(datasetManager, options = {}, progressCallback = null) {
    if (!datasetManager.isLoaded) {
      await datasetManager.load();
    }

    const results = {
      timestamp: new Date().toISOString(),
      datasetName: 'Wisconsin Diagnostic Breast Cancer (WDBC)',
      totalSamples: datasetManager.stats.totalSamples,
      originalDimensions: datasetManager.stats.numFeatures,
      classDistribution: datasetManager.stats.classDistribution,
      algorithms: {},
      evaluations: {},
      comparisonTable: []
    };

    // 1. Baseline Evaluation on Original 30 Features
    if (progressCallback) progressCallback(10, 'Evaluating Original 30 Features...');
    const originalMetrics = ClassifierEvaluator.evaluateFeatureSet(
      datasetManager.X_std,
      datasetManager.y,
      datasetManager.trainTestSplit
    );
    results.evaluations['Original (30D)'] = originalMetrics;

    // 2. PCA Execution
    if (progressCallback) progressCallback(25, 'Computing Principal Component Analysis...');
    const pcaRes = PCARunner.run(datasetManager.X_std, options.pcaComponents || 2);
    results.algorithms['PCA'] = pcaRes;

    const pcaMetrics = ClassifierEvaluator.evaluateFeatureSet(
      pcaRes.projections,
      datasetManager.y,
      datasetManager.trainTestSplit
    );
    results.evaluations['PCA'] = pcaMetrics;

    // 3. SVD Execution
    if (progressCallback) progressCallback(45, 'Computing Singular Value Decomposition...');
    const svdRes = SVDRunner.run(datasetManager.X_std, options.svdComponents || 2);
    results.algorithms['SVD'] = svdRes;

    const svdMetrics = ClassifierEvaluator.evaluateFeatureSet(
      svdRes.projections,
      datasetManager.y,
      datasetManager.trainTestSplit
    );
    results.evaluations['SVD'] = svdMetrics;

    // 4. t-SNE Execution
    if (progressCallback) progressCallback(65, 'Running t-SNE non-linear embedding...');
    // We can use all samples or subsample for speed if selected
    const tsneSample = options.useSubsample ? datasetManager.getSubsample(options.subsampleSize || 250) : null;
    const tsneData = tsneSample ? tsneSample.X_std : datasetManager.X_std;
    const tsneY = tsneSample ? tsneSample.y : datasetManager.y;

    const tsneRes = await TSNERunner.run(tsneData, {
      dim: options.tsneDim || 2,
      perplexity: options.perplexity || 30,
      iterations: options.tsneIterations || 200
    });
    results.algorithms['t-SNE'] = tsneRes;

    // Standardize t-SNE embeddings for linear classifier
    const tsneStd = MathLib.project(tsneRes.projections, [[1, 0], [0, 1]]);
    // Use matching train/test split
    const tsneLen = tsneY.length;
    const trainCountTsne = Math.floor(tsneLen * 0.7);
    const splitForTsne = tsneSample
      ? {
          trainIndices: Array.from({ length: trainCountTsne }, (_, i) => i),
          testIndices: Array.from({ length: tsneLen - trainCountTsne }, (_, i) => trainCountTsne + i)
        }
      : datasetManager.trainTestSplit;

    const tsneMetrics = ClassifierEvaluator.evaluateFeatureSet(tsneRes.projections, tsneY, splitForTsne);
    results.evaluations['t-SNE'] = tsneMetrics;

    // 5. UMAP Execution
    if (progressCallback) progressCallback(85, 'Running UMAP manifold learning...');
    const umapSample = options.useSubsample ? datasetManager.getSubsample(options.subsampleSize || 250) : null;
    const umapData = umapSample ? umapSample.X_std : datasetManager.X_std;
    const umapY = umapSample ? umapSample.y : datasetManager.y;

    const umapRes = await UMAPRunner.run(umapData, {
      nComponents: options.umapComponents || 2,
      nNeighbors: options.nNeighbors || 15,
      minDist: options.minDist !== undefined ? options.minDist : 0.1
    });
    results.algorithms['UMAP'] = umapRes;

    const umapLen = umapY.length;
    const trainCountUmap = Math.floor(umapLen * 0.7);
    const splitForUmap = umapSample
      ? {
          trainIndices: Array.from({ length: trainCountUmap }, (_, i) => i),
          testIndices: Array.from({ length: umapLen - trainCountUmap }, (_, i) => trainCountUmap + i)
        }
      : datasetManager.trainTestSplit;

    const umapMetrics = ClassifierEvaluator.evaluateFeatureSet(umapRes.projections, umapY, splitForUmap);
    results.evaluations['UMAP'] = umapMetrics;

    if (progressCallback) progressCallback(100, 'All benchmarks completed successfully!');

    // Build consolidated summary comparison table
    results.comparisonTable = [
      {
        technique: 'Original Features',
        type: 'High-Dimensional Baseline',
        reducedDim: 30,
        explainedVariance: '100%',
        runtimeMs: originalMetrics.evalTimeMs,
        accuracy: originalMetrics.accuracy,
        precision: originalMetrics.precision,
        recall: originalMetrics.recall,
        f1Score: originalMetrics.f1
      },
      {
        technique: 'PCA',
        type: 'Linear / Global Variance',
        reducedDim: pcaRes.nComponents,
        explainedVariance: pcaRes.totalExplainedVariance + '%',
        runtimeMs: pcaRes.executionTimeMs,
        accuracy: pcaMetrics.accuracy,
        precision: pcaMetrics.precision,
        recall: pcaMetrics.recall,
        f1Score: pcaMetrics.f1
      },
      {
        technique: 'SVD',
        type: 'Linear / Matrix Factorization',
        reducedDim: svdRes.nComponents,
        explainedVariance: svdRes.totalExplainedVariance + '%',
        runtimeMs: svdRes.executionTimeMs,
        accuracy: svdMetrics.accuracy,
        precision: svdMetrics.precision,
        recall: svdMetrics.recall,
        f1Score: svdMetrics.f1
      },
      {
        technique: 't-SNE',
        type: 'Non-Linear / Local Neighborhoods',
        reducedDim: tsneRes.nComponents,
        explainedVariance: 'N/A (Probabilistic Divergence)',
        runtimeMs: tsneRes.executionTimeMs,
        accuracy: tsneMetrics.accuracy,
        precision: tsneMetrics.precision,
        recall: tsneMetrics.recall,
        f1Score: tsneMetrics.f1
      },
      {
        technique: 'UMAP',
        type: 'Non-Linear / Fuzzy Simplicial Set',
        reducedDim: umapRes.nComponents,
        explainedVariance: 'N/A (Fuzzy Cross-Entropy)',
        runtimeMs: umapRes.executionTimeMs,
        accuracy: umapMetrics.accuracy,
        precision: umapMetrics.precision,
        recall: umapMetrics.recall,
        f1Score: umapMetrics.f1
      }
    ];

    // Persist in localStorage so report.html can access exact experiment runs
    try {
      localStorage.setItem('vlab_experiment_results', JSON.stringify(results));
    } catch (e) {
      console.warn('Could not persist experiment results to localStorage:', e);
    }

    return results;
  }
}

// ==========================================
// 8. DATA EXPORT UTILITIES (CSV)
// ==========================================

class Exporter {
  static downloadCSV(filename, csvContent) {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  static exportComparisonTable(comparisonTable) {
    const headers = ['Technique', 'Type', 'Reduced_Dimensions', 'Explained_Variance', 'Runtime_ms', 'Accuracy_pct', 'Precision_pct', 'Recall_pct', 'F1_Score_pct'];
    const rows = comparisonTable.map(r => [
      r.technique,
      r.type,
      r.reducedDim,
      `"${r.explainedVariance}"`,
      r.runtimeMs,
      r.accuracy,
      r.precision,
      r.recall,
      r.f1Score
    ].join(','));

    const csv = [headers.join(','), ...rows].join('\n');
    this.downloadCSV('dimensionality_reduction_comparison.csv', csv);
  }

  static exportEmbeddings(algorithmName, projections, labels, ids) {
    const is3D = projections[0].length === 3;
    const headers = is3D
      ? ['sample_id', 'diagnosis', 'component_1', 'component_2', 'component_3']
      : ['sample_id', 'diagnosis', 'component_1', 'component_2'];

    const rows = projections.map((coords, i) => {
      const id = ids[i] || `Sample_${i + 1}`;
      const diag = labels[i] || 'Unknown';
      return is3D
        ? [id, diag, coords[0].toFixed(6), coords[1].toFixed(6), coords[2].toFixed(6)].join(',')
        : [id, diag, coords[0].toFixed(6), coords[1].toFixed(6)].join(',');
    });

    const csv = [headers.join(','), ...rows].join('\n');
    this.downloadCSV(`${algorithmName.toLowerCase()}_reduced_coordinates.csv`, csv);
  }
}

// Expose globally
window.MathLib = MathLib;
window.PCARunner = PCARunner;
window.SVDRunner = SVDRunner;
window.TSNERunner = TSNERunner;
window.UMAPRunner = UMAPRunner;
window.ClassifierEvaluator = ClassifierEvaluator;
window.ExperimentSuite = ExperimentSuite;
window.Exporter = Exporter;
