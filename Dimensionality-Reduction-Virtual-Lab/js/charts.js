/**
 * Dimensionality Reduction Virtual Lab
 * Visualization Module using Plotly.js
 */

const ChartTheme = {
  paper_bgcolor: 'rgba(15, 23, 42, 0)', // Transparent over dark cards
  plot_bgcolor: 'rgba(11, 17, 32, 0.65)',
  font: {
    family: 'Inter, system-ui, -apple-system, sans-serif',
    color: '#cbd5e1',
    size: 12
  },
  gridcolor: 'rgba(51, 65, 85, 0.45)',
  zerolinecolor: 'rgba(99, 102, 241, 0.35)',
  colorBenign: '#10b981', // Emerald green
  colorMalignant: '#f43f5e', // Rose red
  colorPC1: '#6366f1', // Indigo
  colorPC2: '#8b5cf6', // Violet
  colorCumulative: '#06b6d4', // Cyan
  accent: '#a855f7'
};

class ChartManager {
  /**
   * Renders a 2D or 3D Scatter Plot for an algorithm projection
   */
  static renderScatterPlot(containerId, algorithmName, projections, labels, ids, nComponents = 2) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const is3D = nComponents === 3 && projections[0] && projections[0].length >= 3;

    // Separate Benign and Malignant points
    const benign = { x: [], y: [], z: [], ids: [] };
    const malignant = { x: [], y: [], z: [], ids: [] };

    for (let i = 0; i < projections.length; i++) {
      const isM = labels[i] === 'Malignant' || labels[i] === 'M' || labels[i] === 1;
      const target = isM ? malignant : benign;
      const row = projections[i];
      target.x.push(row[0]);
      target.y.push(row[1]);
      if (is3D) target.z.push(row[2]);
      target.ids.push(ids[i] || `Sample_${i + 1}`);
    }

    const traces = [];

    if (is3D) {
      // 3D Scatter Plot
      traces.push({
        name: `Benign (${benign.x.length})`,
        x: benign.x,
        y: benign.y,
        z: benign.z,
        mode: 'markers',
        type: 'scatter3d',
        customdata: benign.ids.map(id => [id, 'Benign']),
        hovertemplate: '<b>Sample:</b> %{customdata[0]}<br><b>Class:</b> %{customdata[1]}<br>C1: %{x:.2f}<br>C2: %{y:.2f}<br>C3: %{z:.2f}<extra></extra>',
        marker: {
          size: 4.5,
          color: ChartTheme.colorBenign,
          opacity: 0.85
        }
      });

      traces.push({
        name: `Malignant (${malignant.x.length})`,
        x: malignant.x,
        y: malignant.y,
        z: malignant.z,
        mode: 'markers',
        type: 'scatter3d',
        customdata: malignant.ids.map(id => [id, 'Malignant']),
        hovertemplate: '<b>Sample:</b> %{customdata[0]}<br><b>Class:</b> %{customdata[1]}<br>C1: %{x:.2f}<br>C2: %{y:.2f}<br>C3: %{z:.2f}<extra></extra>',
        marker: {
          size: 4.5,
          color: ChartTheme.colorMalignant,
          opacity: 0.85
        }
      });

      const layout = {
        title: {
          text: `<b>${algorithmName}</b> 3D Component Embedding (WDBC)`,
          font: { size: 14, color: '#f8fafc' },
          x: 0.05
        },
        paper_bgcolor: ChartTheme.paper_bgcolor,
        font: ChartTheme.font,
        scene: {
          xaxis: { title: 'Component 1', gridcolor: ChartTheme.gridcolor, backgroundcolor: 'rgba(15, 23, 42, 0.4)' },
          yaxis: { title: 'Component 2', gridcolor: ChartTheme.gridcolor, backgroundcolor: 'rgba(15, 23, 42, 0.4)' },
          zaxis: { title: 'Component 3', gridcolor: ChartTheme.gridcolor, backgroundcolor: 'rgba(15, 23, 42, 0.4)' },
          camera: {
            eye: { x: 1.4, y: 1.4, z: 1.1 }
          }
        },
        legend: {
          orientation: 'h',
          y: -0.1,
          x: 0.5,
          xanchor: 'center',
          font: { color: '#e2e8f0' }
        },
        margin: { l: 20, r: 20, b: 40, t: 50 }
      };

      Plotly.react(containerId, traces, layout, { responsive: true, displayModeBar: true, displaylogo: false });
    } else {
      // 2D Scatter Plot
      traces.push({
        name: `Benign (${benign.x.length})`,
        x: benign.x,
        y: benign.y,
        mode: 'markers',
        type: 'scatter',
        customdata: benign.ids.map(id => [id, 'Benign']),
        hovertemplate: '<b>Sample:</b> %{customdata[0]}<br><b>Class:</b> %{customdata[1]}<br>Dim 1: %{x:.2f}<br>Dim 2: %{y:.2f}<extra></extra>',
        marker: {
          size: 7.5,
          color: ChartTheme.colorBenign,
          opacity: 0.85,
          line: { color: '#064e3b', width: 0.5 }
        }
      });

      traces.push({
        name: `Malignant (${malignant.x.length})`,
        x: malignant.x,
        y: malignant.y,
        mode: 'markers',
        type: 'scatter',
        customdata: malignant.ids.map(id => [id, 'Malignant']),
        hovertemplate: '<b>Sample:</b> %{customdata[0]}<br><b>Class:</b> %{customdata[1]}<br>Dim 1: %{x:.2f}<br>Dim 2: %{y:.2f}<extra></extra>',
        marker: {
          size: 7.5,
          color: ChartTheme.colorMalignant,
          opacity: 0.85,
          symbol: 'diamond',
          line: { color: '#4c0519', width: 0.5 }
        }
      });

      const layout = {
        title: {
          text: `<b>${algorithmName}</b> 2D Projection (Benign vs Malignant)`,
          font: { size: 14, color: '#f8fafc' },
          x: 0.05
        },
        paper_bgcolor: ChartTheme.paper_bgcolor,
        plot_bgcolor: ChartTheme.plot_bgcolor,
        font: ChartTheme.font,
        xaxis: {
          title: algorithmName === 'PCA' ? 'Principal Component 1 (PC1)' : (algorithmName === 'SVD' ? 'Singular Vector 1' : 'Dimension 1'),
          gridcolor: ChartTheme.gridcolor,
          zerolinecolor: ChartTheme.zerolinecolor,
          zeroline: true
        },
        yaxis: {
          title: algorithmName === 'PCA' ? 'Principal Component 2 (PC2)' : (algorithmName === 'SVD' ? 'Singular Vector 2' : 'Dimension 2'),
          gridcolor: ChartTheme.gridcolor,
          zerolinecolor: ChartTheme.zerolinecolor,
          zeroline: true
        },
        legend: {
          orientation: 'h',
          y: 1.12,
          x: 0.98,
          xanchor: 'right',
          font: { color: '#e2e8f0' }
        },
        margin: { l: 55, r: 25, b: 50, t: 60 }
      };

      Plotly.react(containerId, traces, layout, { responsive: true, displayModeBar: true, displaylogo: false });
    }
  }

  /**
   * Renders PCA Scree & Cumulative Variance Chart
   */
  static renderPCAScree(containerId, explainedVarianceRatios, cumulativeVariances) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const numComponents = Math.min(10, explainedVarianceRatios.length);
    const labels = Array.from({ length: numComponents }, (_, i) => `PC${i + 1}`);
    const individual = explainedVarianceRatios.slice(0, numComponents);
    const cumulative = cumulativeVariances.slice(0, numComponents);

    const traceBar = {
      name: 'Individual Variance',
      x: labels,
      y: individual,
      type: 'bar',
      marker: {
        color: '#6366f1',
        opacity: 0.85
      },
      hovertemplate: '%{x}: %{y:.2f}% variance<extra></extra>'
    };

    const traceLine = {
      name: 'Cumulative Variance',
      x: labels,
      y: cumulative,
      type: 'scatter',
      mode: 'lines+markers',
      yaxis: 'y2',
      line: {
        color: '#06b6d4',
        width: 2.5
      },
      marker: {
        color: '#22d3ee',
        size: 7
      },
      hovertemplate: '%{x} Cumulative: %{y:.2f}%<extra></extra>'
    };

    const layout = {
      title: {
        text: '<b>PCA Scree Plot</b>: Explained & Cumulative Variance',
        font: { size: 14, color: '#f8fafc' },
        x: 0.05
      },
      paper_bgcolor: ChartTheme.paper_bgcolor,
      plot_bgcolor: ChartTheme.plot_bgcolor,
      font: ChartTheme.font,
      xaxis: {
        title: 'Principal Components',
        gridcolor: ChartTheme.gridcolor
      },
      yaxis: {
        title: 'Individual Variance (%)',
        gridcolor: ChartTheme.gridcolor,
        range: [0, Math.max(...individual) * 1.15]
      },
      yaxis2: {
        title: 'Cumulative (%)',
        overlaying: 'y',
        side: 'right',
        showgrid: false,
        range: [0, 105],
        tickfont: { color: '#06b6d4' }
      },
      legend: {
        orientation: 'h',
        y: 1.14,
        x: 0.98,
        xanchor: 'right'
      },
      margin: { l: 55, r: 55, b: 50, t: 60 }
    };

    Plotly.react(containerId, [traceBar, traceLine], layout, { responsive: true, displayModeBar: false });
  }

  /**
   * Renders PCA Feature Importance / Loadings Horizontal Bar Chart
   */
  static renderFeatureImportance(containerId, featureContributions, topK = 12) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const topFeatures = featureContributions.slice(0, topK).reverse();
    const names = topFeatures.map(f => f.label || f.name);
    const pc1Weights = topFeatures.map(f => Math.abs(f.pc1Weight));
    const pc2Weights = topFeatures.map(f => Math.abs(f.pc2Weight));

    const tracePC1 = {
      name: 'PC1 Weight (|loading|)',
      y: names,
      x: pc1Weights,
      type: 'bar',
      orientation: 'h',
      marker: {
        color: '#6366f1',
        opacity: 0.9
      },
      hovertemplate: '%{y} (PC1): %{x:.4f}<extra></extra>'
    };

    const tracePC2 = {
      name: 'PC2 Weight (|loading|)',
      y: names,
      x: pc2Weights,
      type: 'bar',
      orientation: 'h',
      marker: {
        color: '#a855f7',
        opacity: 0.8
      },
      hovertemplate: '%{y} (PC2): %{x:.4f}<extra></extra>'
    };

    const layout = {
      title: {
        text: `<b>Top ${topK} Features</b> Contributing to Principal Components`,
        font: { size: 14, color: '#f8fafc' },
        x: 0.05
      },
      barmode: 'group',
      paper_bgcolor: ChartTheme.paper_bgcolor,
      plot_bgcolor: ChartTheme.plot_bgcolor,
      font: ChartTheme.font,
      xaxis: {
        title: 'Absolute Eigenvector Loading Weight',
        gridcolor: ChartTheme.gridcolor
      },
      yaxis: {
        automargin: true,
        gridcolor: 'transparent'
      },
      legend: {
        orientation: 'h',
        y: 1.12,
        x: 0.98,
        xanchor: 'right'
      },
      margin: { l: 150, r: 30, b: 50, t: 60 }
    };

    Plotly.react(containerId, [tracePC1, tracePC2], layout, { responsive: true, displayModeBar: false });
  }

  /**
   * Renders Classification Performance Comparison Grouped Bar Chart
   */
  static renderModelComparisonChart(containerId, evaluations) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const techniques = Object.keys(evaluations);
    const accuracies = techniques.map(t => evaluations[t].accuracy);
    const f1Scores = techniques.map(t => evaluations[t].f1);
    const precisions = techniques.map(t => evaluations[t].precision);
    const recalls = techniques.map(t => evaluations[t].recall);

    const traceAcc = {
      name: 'Accuracy (%)',
      x: techniques,
      y: accuracies,
      type: 'bar',
      marker: { color: '#6366f1' },
      hovertemplate: '%{x} Accuracy: %{y:.2f}%<extra></extra>'
    };

    const traceF1 = {
      name: 'F1-Score (%)',
      x: techniques,
      y: f1Scores,
      type: 'bar',
      marker: { color: '#10b981' },
      hovertemplate: '%{x} F1-Score: %{y:.2f}%<extra></extra>'
    };

    const tracePrec = {
      name: 'Precision (%)',
      x: techniques,
      y: precisions,
      type: 'bar',
      marker: { color: '#38bdf8' },
      hovertemplate: '%{x} Precision: %{y:.2f}%<extra></extra>'
    };

    const layout = {
      title: {
        text: '<b>Classification Performance Comparison</b> (Logistic Regression)',
        font: { size: 14, color: '#f8fafc' },
        x: 0.05
      },
      barmode: 'group',
      paper_bgcolor: ChartTheme.paper_bgcolor,
      plot_bgcolor: ChartTheme.plot_bgcolor,
      font: ChartTheme.font,
      xaxis: {
        title: 'Feature Representation',
        gridcolor: ChartTheme.gridcolor
      },
      yaxis: {
        title: 'Percentage Score (%)',
        gridcolor: ChartTheme.gridcolor,
        range: [70, 102]
      },
      legend: {
        orientation: 'h',
        y: 1.14,
        x: 0.98,
        xanchor: 'right'
      },
      margin: { l: 55, r: 25, b: 50, t: 60 }
    };

    Plotly.react(containerId, [traceAcc, traceF1, tracePrec], layout, { responsive: true, displayModeBar: false });
  }

  /**
   * Renders mini 2D scatter plots for side-by-side comparison grid
   */
  static renderMiniScatter(containerId, title, projections, labels, ids) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const benign = { x: [], y: [] };
    const malignant = { x: [], y: [] };

    for (let i = 0; i < projections.length; i++) {
      const isM = labels[i] === 'Malignant' || labels[i] === 'M' || labels[i] === 1;
      const target = isM ? malignant : benign;
      target.x.push(projections[i][0]);
      target.y.push(projections[i][1]);
    }

    const traces = [
      {
        name: 'Benign',
        x: benign.x,
        y: benign.y,
        mode: 'markers',
        type: 'scatter',
        marker: { size: 5, color: ChartTheme.colorBenign, opacity: 0.8 },
        hoverinfo: 'none'
      },
      {
        name: 'Malignant',
        x: malignant.x,
        y: malignant.y,
        mode: 'markers',
        type: 'scatter',
        marker: { size: 5, color: ChartTheme.colorMalignant, opacity: 0.8 },
        hoverinfo: 'none'
      }
    ];

    const layout = {
      title: {
        text: `<b>${title}</b>`,
        font: { size: 12, color: '#f8fafc' },
        x: 0.05
      },
      showlegend: false,
      paper_bgcolor: ChartTheme.paper_bgcolor,
      plot_bgcolor: ChartTheme.plot_bgcolor,
      font: { size: 10, color: '#94a3b8' },
      xaxis: { showticklabels: false, gridcolor: ChartTheme.gridcolor },
      yaxis: { showticklabels: false, gridcolor: ChartTheme.gridcolor },
      margin: { l: 20, r: 20, b: 20, t: 35 }
    };

    Plotly.react(containerId, traces, layout, { responsive: true, displayModeBar: false });
  }
}

// Expose globally
window.ChartManager = ChartManager;
window.ChartTheme = ChartTheme;

