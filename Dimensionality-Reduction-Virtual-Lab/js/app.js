/**
 * Dimensionality Reduction Virtual Lab
 * Application Controller & Event Orchestrator
 */

// Application State
const AppState = {
  currentTab: 'PCA',
  activeResults: {
    PCA: null,
    SVD: null,
    't-SNE': null,
    UMAP: null
  },
  lastFullSuite: null,
  isProcessing: false
};

// Interpretations Database
const Interpretations = {
  PCA: (varPct, nComp) => `
    <strong>Principal Component Analysis (PCA)</strong> is a linear orthogonal transformation. 
    The top ${nComp} principal components capture <strong>${varPct}%</strong> of the total variance across all 30 cellular characteristics. 
    Notice the clear linear separation trend between Benign (green) and Malignant (red) classes along PC1. 
    Features associated with cell nuclear size and irregularity (such as <em>worst perimeter</em>, <em>worst radius</em>, and <em>worst concave points</em>) 
    have the strongest loadings along PC1, explaining why high PC1 values correlate strongly with malignancy.
  `,
  SVD: (varPct, nComp) => `
    <strong>Singular Value Decomposition (SVD)</strong> decomposes the standardized feature matrix into singular vectors and singular values. 
    Because data standardization was applied, the right singular vectors correspond mathematically to the principal axes, capturing 
    <strong>${varPct}%</strong> of the matrix energy. SVD preserves global geometry and pairwise Euclidean distances without requiring non-linear projections, 
    making it computationally instantaneous while providing strong class separability.
  `,
  't-SNE': (perplexity, nComp) => `
    <strong>t-Distributed Stochastic Neighbor Embedding (t-SNE)</strong> is a non-linear probabilistic technique that preserves <em>local neighborhoods</em>. 
    At perplexity = ${perplexity}, local manifold clusters are highlighted. Malignant tumors cluster into distinct islands with tighter group cohesion, 
    revealing sub-types and borderline lesions that overlap less than in linear PCA. Note that distances between distant clusters in t-SNE do not reflect 
    global Euclidean distances.
  `,
  UMAP: (nNeighbors, minDist, nComp) => `
    <strong>Uniform Manifold Approximation and Projection (UMAP)</strong> combines Riemannian geometry with fuzzy simplicial sets. 
    With ${nNeighbors} nearest neighbors and minimum distance ${minDist}, UMAP preserves both <em>local clustering</em> and <em>global inter-cluster hierarchy</em>. 
    The Benign and Malignant distributions form distinct manifolds with clear decision boundaries, explaining why non-linear classifiers achieve high accuracy 
    on UMAP embeddings.
  `
};

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  setupMobileMenu();
  setupTabSwitching();
  setupEventListeners();

  try {
    // Load WDBC dataset
    updateDatasetStatus('Loading dataset...');
    await window.datasetManager.load();
    updateDatasetStatus('WDBC Loaded (569 samples)');
    populateDatasetStats(window.datasetManager.stats);

    // Run initial PCA to give user immediate interactive chart
    await runSingleAlgorithm('PCA');

    // Run full benchmark suite in background to populate results table & comparison matrix
    setTimeout(() => {
      runFullExperimentSuite(false);
    }, 400);

  } catch (err) {
    console.error('Initialization error:', err);
    updateDatasetStatus('Error loading dataset');
    alert('Failed to load dataset: ' + err.message);
  }
});

// ==========================================
// NAVIGATION & UI HELPERS
// ==========================================

function setupNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.lab-section');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        navLinks.forEach(link => {
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      }
    });
  }, { threshold: 0.25 });

  sections.forEach(sec => observer.observe(sec));
}

function setupMobileMenu() {
  const mobileToggle = document.getElementById('mobileMenuToggle');
  const sidebar = document.getElementById('appSidebar');

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    // Close when clicking nav links on mobile
    sidebar.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        sidebar.classList.remove('open');
      });
    });
  }
}

function updateDatasetStatus(statusText) {
  const badge = document.getElementById('datasetStatusBadge');
  if (badge) {
    badge.innerHTML = `<span class="status-dot"></span> ${statusText}`;
  }
}

function populateDatasetStats(stats) {
  const elSamples = document.getElementById('statTotalSamples');
  const elFeatures = document.getElementById('statNumFeatures');
  const elMalignant = document.getElementById('statMalignant');
  const elBenign = document.getElementById('statBenign');
  const elMissing = document.getElementById('statMissing');

  if (elSamples) elSamples.textContent = stats.totalSamples;
  if (elFeatures) elFeatures.textContent = stats.numFeatures;
  if (elMalignant) elMalignant.textContent = `${stats.classDistribution.malignant.count} (${stats.classDistribution.malignant.percentage}%)`;
  if (elBenign) elBenign.textContent = `${stats.classDistribution.benign.count} (${stats.classDistribution.benign.percentage}%)`;
  if (elMissing) elMissing.textContent = stats.missingValuesCount;
}

// ==========================================
// SIMULATION TABS & CONTROLS
// ==========================================

function setupTabSwitching() {
  const tabBtns = document.querySelectorAll('.sim-tab-btn');
  const panels = document.querySelectorAll('.tab-params-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetAlgo = btn.getAttribute('data-algo');
      AppState.currentTab = targetAlgo;

      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      panels.forEach(p => {
        p.style.display = p.getAttribute('data-panel') === targetAlgo ? 'flex' : 'none';
      });

      // Show/hide secondary scree charts for PCA
      const screeRow = document.getElementById('pcaSubChartsRow');
      if (screeRow) {
        screeRow.style.display = targetAlgo === 'PCA' ? 'grid' : 'none';
      }

      // If results already exist for this tab, re-render
      if (AppState.activeResults[targetAlgo]) {
        renderAlgorithmOutput(AppState.activeResults[targetAlgo]);
      } else {
        runSingleAlgorithm(targetAlgo);
      }
    });
  });
}

function setupEventListeners() {
  // Run current simulation button
  const runBtn = document.getElementById('btnRunSimulation');
  if (runBtn) {
    runBtn.addEventListener('click', () => {
      runSingleAlgorithm(AppState.currentTab);
    });
  }

  // Reset button
  const resetBtn = document.getElementById('btnResetParams');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      resetTabParameters(AppState.currentTab);
    });
  }

  // Run all 4 algorithms button
  const runAllBtn = document.getElementById('btnRunAllComparison');
  if (runAllBtn) {
    runAllBtn.addEventListener('click', () => {
      runFullExperimentSuite(true);
    });
  }

  // Export buttons
  const btnExportCSV = document.getElementById('btnExportCSV');
  if (btnExportCSV) {
    btnExportCSV.addEventListener('click', () => {
      if (AppState.lastFullSuite && AppState.lastFullSuite.comparisonTable) {
        Exporter.exportComparisonTable(AppState.lastFullSuite.comparisonTable);
      } else {
        alert('Please run the comparison benchmark first to generate results.');
      }
    });
  }

  const btnExportEmbeddings = document.getElementById('btnExportEmbeddings');
  if (btnExportEmbeddings) {
    btnExportEmbeddings.addEventListener('click', () => {
      const active = AppState.activeResults[AppState.currentTab];
      if (active && active.projections) {
        Exporter.exportEmbeddings(active.algorithm, active.projections, window.datasetManager.labels, window.datasetManager.ids);
      } else {
        alert(`No coordinates available for ${AppState.currentTab}. Please run simulation first.`);
      }
    });
  }

  // Sliders with value tags
  bindSliderToTag('tsnePerplexity', 'tsnePerplexityVal');
  bindSliderToTag('tsneIterations', 'tsneIterationsVal');
  bindSliderToTag('umapNeighbors', 'umapNeighborsVal');
  bindSliderToTag('umapMinDist', 'umapMinDistVal', val => parseFloat(val).toFixed(2));
}

function bindSliderToTag(sliderId, tagId, formatter = val => val) {
  const slider = document.getElementById(sliderId);
  const tag = document.getElementById(tagId);
  if (slider && tag) {
    slider.addEventListener('input', () => {
      tag.textContent = formatter(slider.value);
    });
  }
}

function resetTabParameters(algo) {
  if (algo === 'PCA') {
    document.getElementById('pcaComponents').value = '2';
  } else if (algo === 'SVD') {
    document.getElementById('svdComponents').value = '2';
  } else if (algo === 't-SNE') {
    document.getElementById('tsneComponents').value = '2';
    document.getElementById('tsnePerplexity').value = '30';
    document.getElementById('tsnePerplexityVal').textContent = '30';
    document.getElementById('tsneIterations').value = '200';
    document.getElementById('tsneIterationsVal').textContent = '200';
    document.getElementById('tsneSampleSize').value = 'all';
  } else if (algo === 'UMAP') {
    document.getElementById('umapComponents').value = '2';
    document.getElementById('umapNeighbors').value = '15';
    document.getElementById('umapNeighborsVal').textContent = '15';
    document.getElementById('umapMinDist').value = '0.1';
    document.getElementById('umapMinDistVal').textContent = '0.10';
    document.getElementById('umapSampleSize').value = 'all';
  }
  runSingleAlgorithm(algo);
}

// ==========================================
// SIMULATION EXECUTION
// ==========================================

async function runSingleAlgorithm(algo) {
  if (AppState.isProcessing) return;
  AppState.isProcessing = true;

  showProgress(true, `Running ${algo} simulation...`, 20);
  const runBtn = document.getElementById('btnRunSimulation');
  if (runBtn) runBtn.disabled = true;

  try {
    let result = null;
    const dm = window.datasetManager;

    if (algo === 'PCA') {
      const nComp = parseInt(document.getElementById('pcaComponents').value, 10) || 2;
      result = PCARunner.run(dm.X_std, nComp);
    } else if (algo === 'SVD') {
      const nComp = parseInt(document.getElementById('svdComponents').value, 10) || 2;
      result = SVDRunner.run(dm.X_std, nComp);
    } else if (algo === 't-SNE') {
      const nComp = parseInt(document.getElementById('tsneComponents').value, 10) || 2;
      const perplexity = parseFloat(document.getElementById('tsnePerplexity').value) || 30;
      const iterations = parseInt(document.getElementById('tsneIterations').value, 10) || 200;
      const sampleMode = document.getElementById('tsneSampleSize').value;

      const dataToUse = sampleMode === '250' ? dm.getSubsample(250) : { X_std: dm.X_std, labels: dm.labels, ids: dm.ids, y: dm.y };

      result = await TSNERunner.run(dataToUse.X_std, {
        dim: nComp,
        perplexity: perplexity,
        iterations: iterations
      }, (pct, iter, total) => {
        showProgress(true, `t-SNE Iteration ${iter}/${total}...`, pct);
      });

      result.labels = dataToUse.labels;
      result.ids = dataToUse.ids;
      result.y = dataToUse.y;
    } else if (algo === 'UMAP') {
      const nComp = parseInt(document.getElementById('umapComponents').value, 10) || 2;
      const nNeighbors = parseInt(document.getElementById('umapNeighbors').value, 10) || 15;
      const minDist = parseFloat(document.getElementById('umapMinDist').value) || 0.1;
      const sampleMode = document.getElementById('umapSampleSize').value;

      const dataToUse = sampleMode === '250' ? dm.getSubsample(250) : { X_std: dm.X_std, labels: dm.labels, ids: dm.ids, y: dm.y };

      showProgress(true, 'Constructing fuzzy simplicial set...', 40);
      result = await UMAPRunner.run(dataToUse.X_std, {
        nComponents: nComp,
        nNeighbors: nNeighbors,
        minDist: minDist
      }, (pct, msg) => {
        showProgress(true, msg, pct);
      });

      result.labels = dataToUse.labels;
      result.ids = dataToUse.ids;
      result.y = dataToUse.y;
    }

    // Evaluate logistic regression on the reduced feature projections
    const testLabels = result.y || dm.y;
    const split = (result.y && result.y.length < dm.y.length)
      ? { trainIndices: Array.from({ length: Math.floor(result.y.length * 0.7) }, (_, i) => i), testIndices: Array.from({ length: Math.ceil(result.y.length * 0.3) }, (_, i) => Math.floor(result.y.length * 0.7) + i) }
      : dm.trainTestSplit;

    const evalMetrics = ClassifierEvaluator.evaluateFeatureSet(result.projections, testLabels, split);
    result.evalMetrics = evalMetrics;

    AppState.activeResults[algo] = result;
    renderAlgorithmOutput(result);

  } catch (error) {
    console.error(`Error executing ${algo}:`, error);
    alert(`Simulation failed for ${algo}: ${error.message}`);
  } finally {
    showProgress(false);
    if (runBtn) runBtn.disabled = false;
    AppState.isProcessing = false;
  }
}

// ==========================================
// RENDERING RESULTS & CHARTS
// ==========================================

function renderAlgorithmOutput(result) {
  const algo = result.algorithm;
  const dm = window.datasetManager;
  const labels = result.labels || dm.labels;
  const ids = result.ids || dm.ids;

  // 1. Update Metrics Ribbon
  const elRuntime = document.getElementById('ribbonRuntime');
  const elDims = document.getElementById('ribbonDims');
  const elVariance = document.getElementById('ribbonVariance');
  const elAccuracy = document.getElementById('ribbonAccuracy');

  if (elRuntime) elRuntime.textContent = `${result.executionTimeMs} ms`;
  if (elDims) elDims.textContent = `30 → ${result.nComponents}D`;
  if (elVariance) {
    elVariance.textContent = result.totalExplainedVariance !== undefined ? `${result.totalExplainedVariance}%` : 'N/A';
  }
  if (elAccuracy && result.evalMetrics) {
    elAccuracy.textContent = `${result.evalMetrics.accuracy}% (F1: ${result.evalMetrics.f1}%)`;
  }

  // 2. Main Scatter Plot
  ChartManager.renderScatterPlot('mainChartCanvas', algo, result.projections, labels, ids, result.nComponents);

  // 3. PCA-Specific Scree and Loadings
  const screeRow = document.getElementById('pcaSubChartsRow');
  if (algo === 'PCA') {
    if (screeRow) screeRow.style.display = 'grid';
    ChartManager.renderPCAScree('pcaScreeCanvas', result.explainedVarianceRatio, result.cumulativeVariance);
    ChartManager.renderFeatureImportance('pcaLoadingsCanvas', result.featureContributions, 10);
  } else {
    if (screeRow) screeRow.style.display = 'none';
  }

  // 4. Interpretation Text
  const interpCard = document.getElementById('graphInterpretationText');
  if (interpCard) {
    if (algo === 'PCA') {
      interpCard.innerHTML = Interpretations.PCA(result.totalExplainedVariance, result.nComponents);
    } else if (algo === 'SVD') {
      interpCard.innerHTML = Interpretations.SVD(result.totalExplainedVariance, result.nComponents);
    } else if (algo === 't-SNE') {
      interpCard.innerHTML = Interpretations['t-SNE'](result.perplexity, result.nComponents);
    } else if (algo === 'UMAP') {
      interpCard.innerHTML = Interpretations.UMAP(result.nNeighbors, result.minDist, result.nComponents);
    }
  }
}

// ==========================================
// FULL EXPERIMENT SUITE & COMPARISON
// ==========================================

async function runFullExperimentSuite(scrollAfter = false) {
  const compareBtn = document.getElementById('btnRunAllComparison');
  if (compareBtn) {
    compareBtn.disabled = true;
    compareBtn.textContent = 'Running Full Suite...';
  }

  showProgress(true, 'Initializing full benchmark suite...', 10);

  try {
    const results = await ExperimentSuite.runFullExperiment(window.datasetManager, {
      pcaComponents: 2,
      svdComponents: 2,
      tsneDim: 2,
      perplexity: 30,
      tsneIterations: 200,
      umapComponents: 2,
      nNeighbors: 15,
      minDist: 0.1,
      useSubsample: false // Full 569 samples for genuine academic precision
    }, (pct, status) => {
      showProgress(true, status, pct);
    });

    AppState.lastFullSuite = results;
    AppState.activeResults['PCA'] = results.algorithms['PCA'];
    AppState.activeResults['SVD'] = results.algorithms['SVD'];
    AppState.activeResults['t-SNE'] = results.algorithms['t-SNE'];
    AppState.activeResults['UMAP'] = results.algorithms['UMAP'];

    // 1. Populate the 4-panel mini comparison grid
    renderComparisonGrid(results);

    // 2. Populate Observations & Results Table
    renderObservationsTable(results.comparisonTable);

    // 3. Populate Comparison Bar Chart
    ChartManager.renderModelComparisonChart('modelComparisonCanvas', results.evaluations);

    // 4. Populate Feature Importance Chart in Observations section
    if (results.algorithms['PCA']) {
      ChartManager.renderFeatureImportance('resultsFeatureImportanceCanvas', results.algorithms['PCA'].featureContributions, 12);
    }

    if (scrollAfter) {
      document.getElementById('observations')?.scrollIntoView({ behavior: 'smooth' });
    }

  } catch (err) {
    console.error('Error running full experiment suite:', err);
    alert('Benchmark failed: ' + err.message);
  } finally {
    showProgress(false);
    if (compareBtn) {
      compareBtn.disabled = false;
      compareBtn.innerHTML = `
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Run All 4 Algorithms & Compare
      `;
    }
  }
}

function renderComparisonGrid(results) {
  const dm = window.datasetManager;
  const algos = ['PCA', 'SVD', 't-SNE', 'UMAP'];

  algos.forEach(algo => {
    const data = results.algorithms[algo];
    if (data && data.projections) {
      ChartManager.renderMiniScatter(`miniChart${algo.replace('-', '')}`, algo, data.projections, dm.labels, dm.ids);

      // Meta values
      const metaRuntime = document.getElementById(`miniMetaRuntime${algo.replace('-', '')}`);
      const metaAcc = document.getElementById(`miniMetaAcc${algo.replace('-', '')}`);
      if (metaRuntime) metaRuntime.textContent = `${data.executionTimeMs} ms`;
      if (metaAcc && results.evaluations[algo]) {
        metaAcc.textContent = `Acc: ${results.evaluations[algo].accuracy}%`;
      }
    }
  });
}

function renderObservationsTable(comparisonTable) {
  const tbody = document.getElementById('resultsTableBody');
  if (!tbody) return;

  tbody.innerHTML = '';
  comparisonTable.forEach(row => {
    const tr = document.createElement('tr');
    if (row.technique === 'Original Features') tr.classList.add('highlight-row');

    tr.innerHTML = `
      <td><strong>${row.technique}</strong></td>
      <td><span class="val-badge ${row.type.includes('Linear') ? 'val-purple' : (row.type.includes('Non-Linear') ? 'val-blue' : 'val-green')}">${row.type}</span></td>
      <td>${row.reducedDim}</td>
      <td>${row.explainedVariance}</td>
      <td>${row.runtimeMs} ms</td>
      <td><strong style="color: #6ee7b7">${row.accuracy}%</strong></td>
      <td>${row.precision}%</td>
      <td>${row.recall}%</td>
      <td><strong style="color: #a5b4fc">${row.f1Score}%</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function showProgress(show, text = 'Processing...', pct = 50) {
  const box = document.getElementById('simProgressBox');
  const txt = document.getElementById('simProgressText');
  const fill = document.getElementById('simProgressFill');

  if (box) box.style.display = show ? 'flex' : 'none';
  if (txt) txt.textContent = text;
  if (fill) fill.style.width = `${pct}%`;
}

