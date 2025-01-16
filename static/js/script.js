"use strict";
// Global Variables
let allColumns = [];
const columnDataStore = {};   // { col: { originalVals, missingVals, kdeX, kdeY, is_numeric, categoryCounts } }
const columnImputations = {}; // { col: { method: distribution[] } }
const columnStats = {};       // { col: { method: { ...stats } } }
const columnInitialized = {}; // tracks if a column’s data has been loaded
const columnPlotData = {};    // stores IDs for charts, forms, etc.
let pipeline = [];            // holds pipeline steps
// imputationReferences holds temporary imputation methods applied per column
const imputationReferences = {}; // { col: [{ method: "method", config: "value" }, ...] }
// Define global arrays to store numeric and categorical column names
let numericColumns = [];
let categoricalColumns = [];

// ------------------------------------------------
// Dataset Imputation (Entire Dataset)
// ------------------------------------------------
document.getElementById("datasetMethod").addEventListener("change", function () {
  const cGroup = document.getElementById("datasetConstantGroup");
  cGroup.style.display = (this.value === "constant") ? "block" : "none";
});

document.getElementById("datasetImputationForm").addEventListener("submit", async function (e) {
  e.preventDefault();
  const method = document.getElementById("datasetMethod").value;
  const cVal = (method === "constant")
    ? document.getElementById("datasetConstantValue").value
    : null;
  try {
    const resp = await fetch("/impute-dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method: method, constant_value: cVal })
    });
    const data = await resp.json();
    if (data.error) {
      alert("Dataset Imputation Error: " + data.error);
    } else {
      alert(data.message);
      updateHeatmap(data.updated_matrix);
    }
  } catch (err) { console.error("Dataset imputation error:", err); }
});

// Utility function to create a safe HTML id
function makeValidId(colName) {
  return colName.replace(/[^a-zA-Z0-9_\-]/g, '_');
}

// Function to populate the dropdown for AI summarization for numeric/categorical columns
function populateStatsTableSelect(columnsArray) {
  const statsSelect = document.getElementById("statsTableSelect");
  if (!statsSelect) {
    console.error("Element with id statsTableSelect not found.");
    return;
  }
  statsSelect.innerHTML = "";
  columnsArray.forEach(col => {
    const option = document.createElement("option");
    option.value = "stats-" + makeValidId(col);
    option.textContent = col;
    statsSelect.appendChild(option);
  });
}

document.getElementById("uploadForm").addEventListener("submit", (e) => {
  e.preventDefault();

  // Show the loading overlay immediately
  showLoading();

  // Record the time when the upload starts
  const startTime = Date.now();

  const formData = new FormData(e.target);
  fetch("/process-dataset", { method: "POST", body: formData })
    .then((r) => r.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        // Hide the overlay even if there is an error
        hideLoading();
        return;
      }

      // Process the returned data (e.g., plot heatmap, build tabs, etc.)
      const { matrix, columns, rows, numeric_cols, categorical_cols } = data;
      plotMissingHeatmap(matrix, columns, rows);
      numericColumns = numeric_cols;
      categoricalColumns = categorical_cols;
      buildColumnTabs(columns);
      populateStatsTableSelect(columns);

      // Calculate elapsed time and ensure a minimum of 1 second delay
      const elapsedTime = Date.now() - startTime;
      const delay = Math.max(0, 1000 - elapsedTime);
      setTimeout(() => { hideLoading(); }, delay);
    })
    .catch(err => {
      console.error("Error uploading CSV:", err);
      hideLoading();
    });
});

document.addEventListener("DOMContentLoaded", () => {
  // Show the greeting overlay when the page loads.
  const greetingOverlay = document.getElementById("greetingOverlay");
  greetingOverlay.style.display = "flex";

  // Hide the greeting overlay when the proceed button is clicked.
  document.getElementById("proceedBtn").addEventListener("click", () => {
    greetingOverlay.style.display = "none";
  });
});

// ------------------------------------------------
// Imputation Existence Check
// ------------------------------------------------
function imputationExists(col, method, constantVal) {
  if (!imputationReferences[col]) return false;
  return imputationReferences[col].some(ref => {
    return ref.method === method && ((method === "constant" && ref.config === constantVal) || method !== "constant");
  });
}

// ------------------------------------------------
// Heatmap Functions
// ------------------------------------------------
function plotMissingHeatmap(matrix, columns, rows) {
  if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(columns) || !Array.isArray(rows)) {
    console.error("Invalid data provided for heatmap:", { matrix, columns, rows });
    return;
  }
  
  // Define a two-step colorscale:
  // 0 maps to transparent light blue; 1 maps to opaque light blue.
  const blueTransparentToOpaque = [
    [0, "rgba(173,216,230,0)"],  // light blue transparent
    [1, "rgb(132, 185, 203)"]   // light blue opaque
  ];
  
  const layout = {
    autosize: true,
    margin: { l: 20, r: 20, t: 20, b: 100 }, // Increased bottom margin for rotated x-axis labels.
    font: { size: 10 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      automargin: true,
      tickangle: -45,
      tickfont: { size: 10 }
    },
    yaxis: {
      automargin: true
    }
  };

  const trace = {
    z: matrix,
    x: columns,
    y: rows,
    type: "heatmap",
    colorscale: blueTransparentToOpaque,
    showscale: false,
    autosize: true
  };

  Plotly.newPlot("missing-heatmap", [trace], layout);
}



function updateHeatmap(newMatrix) {
  Plotly.update("missing-heatmap", { z: [newMatrix] }, {}, [0]);
}

// ------------------------------------------------
// Tabs and Column Content
// ------------------------------------------------
function buildColumnTabs(columns) {
  const tabsList = document.getElementById("columnTabs");
  const tabsContent = document.getElementById("columnTabsContent");
  tabsList.innerHTML = "";
  tabsContent.innerHTML = "";

  columns.forEach((col) => {
    const safeCol = makeValidId(col);
    const tabId = `tab-${safeCol}`;
    const paneId = `pane-${safeCol}`;
    let tabButton = document.createElement("button");
    tabButton.className = "tab-btn";
    tabButton.id = tabId;
    tabButton.textContent = col;
    tabButton.setAttribute("data-target", paneId);
    tabButton.addEventListener("click", () => {
      activateTab(col);
      switchTab(col);
    });
    tabsList.appendChild(tabButton);

    let pane = document.createElement("div");
    pane.className = "tab-pane";
    pane.id = paneId;
    pane.innerHTML = getColumnPaneHTML(col) + `<div id="ref-${safeCol}" class="imputation-ref"></div>`;
    tabsContent.appendChild(pane);

    // Mark column as not initialized yet.
    columnInitialized[col] = false;
  });

  // Activate the first tab by default:
  if (columns.length > 0) {
    switchTab(columns[0]);
  }

  // Pre-load (activate) every column so that their graphs are rendered immediately.
  columns.forEach(col => {
    if (!columnInitialized[col]) {
      activateTab(col);
    }
  });

  // Resize all charts after DOM updates.
  setTimeout(() => resizeAllCharts(), 0);
}

function switchTab(col) {
  const safeCol = makeValidId(col);
  document.querySelectorAll(".tab-pane").forEach((pane) => (pane.style.display = "none"));
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
  const paneToShow = document.getElementById("pane-" + safeCol);
  if (paneToShow) {
    paneToShow.style.display = "block";
  }
  const btn = document.getElementById("tab-" + safeCol);
  if (btn) {
    btn.classList.add("active");
  }
}

function getColumnPaneHTML(col) {
  const safeCol = makeValidId(col);
  // Determine whether the column is numeric via the global array
  const isNumeric = numericColumns.includes(col);

  let chartsHTML = `
    <div class="sub-nav-tabs">
      <button class="active" data-target="sub-hist-${safeCol}">Histogram</button>
      <button data-target="sub-strip-${safeCol}">Missing Bars</button>
  `;

  if (isNumeric) {
    chartsHTML += `
      <button data-target="sub-kde-${safeCol}">KDE</button>
      <button data-target="sub-box-${safeCol}">Box Plot</button>
      <button data-target="sub-corr-${safeCol}">Correlation</button>
    `;
  }

  chartsHTML += `</div>
    <div class="sub-tab-content" id="sub-hist-${safeCol}" style="display: block;">
      <div id="hist-${safeCol}" class="graph" style="height:300px;"></div>
    </div>
    <div class="sub-tab-content" id="sub-strip-${safeCol}" style="display: none;">
      <div id="strip-${safeCol}" class="graph" style="height:300px;"></div>
    </div>
  `;

  if (isNumeric) {
    chartsHTML += `
      <div class="sub-tab-content" id="sub-kde-${safeCol}" style="display: none;">
        <div id="kde-${safeCol}" class="graph" style="height:300px;"></div>
      </div>
      <div class="sub-tab-content" id="sub-box-${safeCol}" style="display: none;">
        <div id="box-${safeCol}" class="graph" style="height:300px;"></div>
      </div>
      <div class="sub-tab-content" id="sub-corr-${safeCol}" style="display: none;">
        <div id="corr-${safeCol}" class="graph" style="height:300px;"></div>
      </div>
    `;
  }

  const formHTML = `
      <form id="form-${safeCol}" class="col-form" data-col="${col}">
        <label for="method-${safeCol}">Method:</label>
        <select id="method-${safeCol}">
          ${isNumeric
      ? `
                <option value="mean">Mean</option>
                <option value="median">Median</option>
                <option value="mode">Mode</option>
                <option value="knn">KNN</option>
                <option value="mice">MICE</option>
                <option value="constant">Constant</option>
                <option value="complete-case">Complete-Case</option>
                <option value="random">Random</option>
              `
      : `
                <option value="mode">Mode</option>
                <option value="constant">Constant</option>
                <option value="complete-case">Complete-Case</option>
                <option value="random">Random</option>
              `}
        </select>
        <div id="cg-${safeCol}" style="display: none;">
          <label for="const-val-${safeCol}">Value:</label>
          <input type="text" id="const-val-${safeCol}" />
        </div>
        <button class="nav-btn" type="submit">Impute</button>
      </form>
      <div class="imputation-reference-container"></div>
      <p>Stats for ${col}</p>
      <table id="stats-${safeCol}">
        <thead>
          <tr>
            <th>Method</th>
            ${isNumeric
      ? `<th>Mean</th>
                 <th>Median</th>
                 <th>Std</th>
                 <th>KDE Overlap</th>
                 <th>Skew</th>
                 <th>Kurt</th>
                 <th>KS Stat</th>
                 <th>KS p-value</th>
                 <th>KL Divergence</th>`
      : `<th>Mode</th>
                 <th>Unique Count</th>
                 <th>Mode Frequency</th>
                 <th>Unique Percentage</th>`
      }
          </tr>
        </thead>
        <tbody></tbody>
      </table>
  `;

  columnPlotData[col] = {
    histogramId: "hist-" + safeCol,
    ...(isNumeric && { kdeId: "kde-" + safeCol, boxId: "box-" + safeCol, corrId: "corr-" + safeCol }),
    stripId: "strip-" + safeCol,
    statsTableId: "stats-" + safeCol,
    formId: "form-" + safeCol,
    methodSelectId: "method-" + safeCol,
    constantGroupId: "cg-" + safeCol,
    constantValueId: "const-val-" + safeCol,
  };

  return `<div class="column-content-container">
            <div class="charts-container">
              ${chartsHTML}
            </div>
            <div class="table-container" id="tableContainer-${safeCol}">
              ${formHTML}
            </div>
          </div>`;
}

function plotCorrelationLine(col) {
  const { corrId } = columnPlotData[col];
  const corrContainer = document.getElementById(corrId);
  if (!corrContainer) {
    console.error(`Correlation plot container with ID ${corrId} not found.`);
    return;
  }
  fetch("/calculate-correlation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column: col }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        console.error(data.error);
        return;
      }
      const trace = {
        x: data.columns,
        y: data.correlations,
        type: "scatter",
        mode: "lines+markers",
        name: "Correlation",
        line: { color: "rgba(49, 98, 157, 0.82)", width: 2 },
      };
      const layout = {
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        autosize: true,
        legend: {
          orientation: "h",
          x: 0.5,
          y: 1.1,
          xanchor: "center",
          yanchor: "bottom",
        },
        margin: { l: 30, r: 30, t: 30, b: 70 },
        font: { size: 10 }
      };
      const config = { responsive: true };
      Plotly.newPlot(corrId, [trace], layout, config)
        .then(() => Plotly.Plots.resize(corrId));
    })
    .catch((err) => console.error("Error fetching correlation data:", err));
}

// ------------------------------------------------
// Prompt Customization and AI Recommendation
// ------------------------------------------------
document.getElementById("customizePromptBtn").addEventListener("click", () => {
  fetch("/get_current_prompt")
    .then(response => response.json())
    .then(data => {
      document.getElementById("customPromptTextArea").value = data.prompt;
      document.getElementById("customizePromptOverlay").style.display = "block";
    })
    .catch(error => console.error("Error fetching prompt:", error));
});
document.getElementById("closePromptOverlay").addEventListener("click", () => {
  document.getElementById("customizePromptOverlay").style.display = "none";
});
document.getElementById("saveCustomPromptBtn").addEventListener("click", () => {
  const updatedPrompt = document.getElementById("customPromptTextArea").value;
  fetch("/update_prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: updatedPrompt })
  })
    .then(response => response.json())
    .then(data => {
      if (data.status === "success") {
        alert("Prompt updated successfully!");
        document.getElementById("customizePromptOverlay").style.display = "none";
      } else {
        alert("Error updating prompt: " + data.message);
      }
    })
    .catch(error => console.error("Error updating prompt:", error));
});
document.getElementById("resetCustomPromptBtn").addEventListener("click", () => {
  fetch("/reset_prompt", { method: "POST" })
    .then(response => response.json())
    .then(data => {
      if (data.status === "success") {
        document.getElementById("customPromptTextArea").value = data.default_prompt;
        alert("Prompt reset to default successfully!");
      } else {
        alert("Error resetting prompt: " + data.message);
      }
    })
    .catch(error => console.error("Error resetting prompt:", error));
});
document.getElementById("recommendBtn").addEventListener("click", async () => {
  // Get the selected stats table to summarize.
  const statsSelect = document.getElementById("statsTableSelect");
  const selectedTableId = statsSelect.value;
  if (!selectedTableId) {
    alert("Please select a column's stats table to summarize.");
    return;
  }
  const statsTableElement = document.getElementById(selectedTableId);
  if (!statsTableElement) {
    alert("Selected stats table not found on the page.");
    return;
  }
  const stats_table_html = statsTableElement.outerHTML;

  // Show the loading overlay
  showLoading();

  try {
    const resp = await fetch("/generate_summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stats_table_html })
    });
    const data = await resp.json();
    // Hide the loading overlay as soon as a response arrives
    hideLoading();

    if (data.success) {
      showOverlay(data.summary);
    } else {
      alert("AI Recommendation Error: " + data.error);
    }
  } catch (err) {
    console.error("Error generating AI recommendation:", err);
    hideLoading();
    alert("Error generating AI recommendation.");
  }
});

function showOverlay(summaryText) {
  document.getElementById("recommendationOutputOverlay").innerText = summaryText;
  document.getElementById("aiOverlay").style.display = "block";
}
document.getElementById("closeOverlay").addEventListener("click", () => {
  document.getElementById("aiOverlay").style.display = "none";
});
window.addEventListener("click", (event) => {
  const overlay = document.getElementById("aiOverlay");
  if (event.target === overlay) {
    overlay.style.display = "none";
  }
});

// ------------------------------------------------
// Sub-tab Switching for Column Charts
// ------------------------------------------------
document.addEventListener("click", (evt) => {
  if (evt.target.matches(".sub-nav-tabs button")) {
    const currentBtn = evt.target;
    const parent = currentBtn.closest(".charts-container");
    const buttons = parent.querySelectorAll(".sub-nav-tabs button");
    const contents = parent.querySelectorAll(".sub-tab-content");
    buttons.forEach(btn => btn.classList.remove("active"));
    contents.forEach(content => content.style.display = "none");
    currentBtn.classList.add("active");
    const targetId = currentBtn.getAttribute("data-target");
    const targetContent = parent.querySelector(`#${targetId}`);
    if (targetContent) {
      targetContent.style.display = "block";
    }
  }
});

// ------------------------------------------------
// Model Selection and AI Model Functions
// ------------------------------------------------
async function loadModels() {
  try {
    const response = await fetch("/get_models");
    const data = await response.json();
    if (data.success) {
      const modelSelect = document.getElementById("modelSelect");
      modelSelect.innerHTML = "";
      data.models.forEach(modelName => {
        const option = document.createElement("option");
        option.value = modelName;
        option.textContent = modelName;
        modelSelect.appendChild(option);
      });
    } else {
      console.error("Error fetching models:", data.error);
    }
  } catch (error) {
    console.error("Error fetching models:", error);
  }
}

async function setModel() {
  const modelSelect = document.getElementById("modelSelect");
  const selectedModel = modelSelect.value;
  if (!selectedModel) {
    alert("Please select a model.");
    return;
  }
  try {
    const formData = new FormData();
    formData.append("model", selectedModel);
    const response = await fetch("/set_model", {
      method: "POST",
      body: formData
    });
    const data = await response.json();
    if (data.success) {
      alert(data.message);
    } else {
      alert("Error setting model: " + data.error);
    }
  } catch (error) {
    console.error("Error setting model:", error);
  }
}
document.getElementById("setModelBtn").addEventListener("click", setModel);
window.addEventListener("DOMContentLoaded", loadModels);

// ------------------------------------------------
// Activate Tab and load data for a column
function activateTab(col) {
  if (columnInitialized[col]) {
    renderImputationReference(col);
    return;
  }
  columnInitialized[col] = true;
  fetchColumnData(col);
  fetchImputationsForColumn(col);
  fetchStatsForColumn(col);
  setupImputationForm(col);
  // Delay calling plotCorrelationLine to allow fetchColumnData to populate columnDataStore
  setTimeout(() => {
    // Only plot correlation if the column is numeric
    if (columnDataStore[col] && columnDataStore[col].is_numeric) {
      plotCorrelationLine(col);
    }
  }, 500);
}

// ------------------------------------------------
// Setup Imputation Form for a Column
// ------------------------------------------------
function setupImputationForm(col) {
  const { formId, methodSelectId, constantGroupId, constantValueId } = columnPlotData[col];
  const formEl = document.getElementById(formId);
  const methodEl = document.getElementById(methodSelectId);
  const constGroup = document.getElementById(constantGroupId);
  const constValueEl = document.getElementById(constantValueId);

  // Toggle constant value input visibility based on selected method
  methodEl.addEventListener("change", () => {
    constGroup.style.display = (methodEl.value === "constant") ? "block" : "none";
  });

  // Form submission with validation for constant value
  formEl.addEventListener("submit", async (evt) => {
    evt.preventDefault();
    const chosenMethod = methodEl.value;
    const cVal = (chosenMethod === "constant") ? constValueEl.value.trim() : null;

    if (chosenMethod === "constant" && (!cVal || cVal === "")) {
      alert("Please provide a constant value for imputation.");
      return;
    }

    if (imputationExists(col, chosenMethod, cVal)) {
      alert(`Imputation "${chosenMethod}" for column "${col}" has already been applied.`);
      return;
    }

    await applyImputation(col, chosenMethod, cVal);
    addImputationReference(col, chosenMethod, cVal);
    addToPipeline(col, chosenMethod, { value: cVal });
  });
}

// ------------------------------------------------
// Fetch Column Data
// ------------------------------------------------
function fetchColumnData(col) {
  fetch("/column-data", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column: col })
  })
    .then(r => r.json())
    .then(d => {
      if (d.error) { console.error(d.error); return; }
      // Attach is_numeric flag and optional categoryCounts from backend if available.
      columnDataStore[col] = {
        originalVals: d.distribution_values || [],
        missingVals: d.missing_values || [],
        kdeX: d.kde_x || [],
        kdeY: d.kde_y || [],
        is_numeric: d.is_numeric,
        categoryCounts: d.category_counts || null
      };
      plotOriginalColumn(col);
    })
    .catch(err => console.error("fetchColumnData error:", err));
}

function plotOriginalColumn(col) {
  const { originalVals, kdeX, kdeY, missingVals, is_numeric, categoryCounts } = columnDataStore[col] || {};
  const { histogramId, stripId } = columnPlotData[col];

  // Plot Histogram / Bar Chart for the original data
  if (originalVals && originalVals.length > 0) {
    let histTrace, layoutConfig = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      autosize: true,
      legend: {
        orientation: "h",
        x: 0.5,
        y: 1.1,
        xanchor: "center",
        yanchor: "bottom"
      },
      margin: { l: 30, r: 30, t: 30, b: 70 },
      font: { size: 10 }
    };
    if (is_numeric) {
      histTrace = {
        x: originalVals,
        type: "histogram",
        name: "Original Histogram",
        marker: { color: "rgba(49, 98, 157, 0.82)" },
        opacity: 0.7,
        autobinx: false,
        xbins: {
          start: Math.min(...originalVals) - 1,
          end: Math.max(...originalVals) + 1,
          size: 1,
        },
      };
      Plotly.newPlot(histogramId, [histTrace], layoutConfig, { responsive: true })
        .then(() => Plotly.Plots.resize(histogramId));
    } else {
      // Use categoryCounts if provided by the backend, otherwise compute counts
      let counts, categories, frequencies;
      if (categoryCounts) {
        counts = categoryCounts;
      } else {
        counts = {};
        originalVals.forEach(val => {
          counts[val] = (counts[val] || 0) + 1;
        });
      }
      categories = Object.keys(counts);
      frequencies = Object.values(counts);
      histTrace = {
        x: categories,
        y: frequencies,
        type: "bar",
        name: "Original Histogram",
        marker: { color: "rgba(49, 98, 157, 0.82)" },
        opacity: 0.7,
      };
      Plotly.newPlot(histogramId, [histTrace], layoutConfig, { responsive: true })
        .then(() => Plotly.Plots.resize(histogramId));
    }
  }

  // Plot Missing Bars (Strip Chart)
  const barTrace = {
    x: missingVals.map((_, i) => i + 1),
    y: missingVals,
    type: "bar",
    name: "Missing Values",
    marker: { color: missingVals.map(v => (v === 1 ? "rgba(157, 49, 99, 0.95)" : "gray")) },
  };
  const barLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    autosize: true,
    legend: {
      orientation: "h",
      x: 0.5,
      y: 1.1,
      xanchor: "center",
      yanchor: "bottom"
    },
    margin: { l: 30, r: 30, t: 30, b: 70 },
    font: { size: 10 }
  };
  Plotly.newPlot(stripId, [barTrace], barLayout, { responsive: true })
    .then(() => Plotly.Plots.resize(stripId));

  if (is_numeric) {
    const { kdeId, boxId } = columnPlotData[col];

    // Plot KDE if available
    if (kdeX && kdeY && kdeX.length > 0 && kdeY.length > 0) {
      const kdeTrace = {
        x: kdeX,
        y: kdeY,
        type: "scatter",
        mode: "lines",
        name: "Original KDE",
        line: { color: "rgba(49, 157, 92, 0.84)", width: 2 },
      };
      const kdeLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        autosize: true,
        legend: {
          orientation: "h",
          x: 0.5,
          y: 1.1,
          xanchor: "center",
          yanchor: "bottom"
        },
        margin: { l: 30, r: 30, t: 30, b: 70 },
        font: { size: 10 }
      };
      Plotly.newPlot(kdeId, [kdeTrace], kdeLayout, { responsive: true })
        .then(() => Plotly.Plots.resize(kdeId));
    }

    // Plot Horizontal Box Plot
    const boxTrace = {
      x: originalVals,
      type: "box",
      orientation: "h",
      name: "Original Box Plot",
      boxpoints: false,
      jitter: 0.3,
      pointpos: -1.8,
      marker: { color: "purple" },
    };
    const boxLayout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      autosize: true,
      legend: {
        orientation: "h",
        x: 0.5,
        y: 1.1,
        xanchor: "center",
        yanchor: "bottom"
      },
      margin: { l: 30, r: 30, t: 30, b: 70 },
      font: { size: 10 },
      yaxis: { showticklabels: false }
    };
    Plotly.newPlot(boxId, [boxTrace], boxLayout, { responsive: true })
      .then(() => Plotly.Plots.resize(boxId));

    // Also refresh the correlation chart if needed
    plotCorrelationLine(col);
  }
}

function resizeAllCharts() {
  document.querySelectorAll(".graph").forEach((container) => {
    if (container.id) Plotly.Plots.resize(container);
  });
}
window.addEventListener("resize", resizeAllCharts);

// ------------------------------------------------
// Fetch Imputations for a Column
// ------------------------------------------------
function fetchImputationsForColumn(col) {
  fetch("/get-imputations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column: col })
  })
    .then(r => r.json())
    .then(d => {
      if (d.error) { console.warn(d.error); return; }
      if (!columnImputations[col]) columnImputations[col] = {};
      Object.entries(d.imputations || {}).forEach(([method, dist]) => {
        columnImputations[col][method] = dist;
        computeLocalKDEandPlot(col, dist, method);
      });
    })
    .catch(err => console.error("fetchImputations err:", err));
}

// ------------------------------------------------
// Fetch Stats for a Column
// ------------------------------------------------
function fetchStatsForColumn(col) {
  fetch("/get-stats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column: col })
  })
    .then(r => r.json())
    .then(d => {
      if (d.error) { console.warn(d.error); return; }
      columnStats[col] = d.stats || {};
      rebuildStatsTable(col);
    })
    .catch(err => console.error("fetchStats err:", err));
}

// ------------------------------------------------
// Rebuild Stats Table for a Column
// ------------------------------------------------
function rebuildStatsTable(col) {
  const statsDict = columnStats[col] || {};
  const tableBody = document.querySelector(`#stats-${makeValidId(col)} tbody`);
  if (!tableBody) {
    console.error("Stats table body not found for column", col);
    return;
  }
  tableBody.innerHTML = "";

  const keys = Object.keys(statsDict);
  if (keys.length === 0) {
    tableBody.innerHTML = "<tr><td>No stats available.</td></tr>";
    return;
  }
  // Use one sample stats object to determine the type of column.
  const sampleStats = statsDict[keys[0]];
  const isNumeric = sampleStats && sampleStats.hasOwnProperty('imputed_mean');

  // Build the Original row.
  const origRow = document.createElement("tr");
  if (isNumeric) {
    origRow.innerHTML = `
      <td><strong>Original</strong></td>
      <td>${toFixedIfNumber(sampleStats.original_mean)}</td>
      <td>${toFixedIfNumber(sampleStats.original_median)}</td>
      <td>${toFixedIfNumber(sampleStats.original_std)}</td>
      <td>—</td>
      <td>${toFixedIfNumber(sampleStats.original_skew)}</td>
      <td>${toFixedIfNumber(sampleStats.original_kurtosis)}</td>
      <td>—</td>
      <td>—</td>
    `;
  } else {
    const orig = sampleStats.original || {};
    origRow.innerHTML = `
      <td><strong>Original</strong></td>
      <td>${orig.mode || "N/A"}</td>
      <td>${orig.unique_count != null ? orig.unique_count : "N/A"}</td>
      <td>${orig.mode_frequency != null ? orig.mode_frequency : "N/A"}</td>
      <td>${orig.unique_percentage != null ? toFixedIfNumber(orig.unique_percentage) + "%" : "N/A"}</td>
    `;
  }
  tableBody.appendChild(origRow);

  // Build rows for each imputation method.
  Object.entries(statsDict).forEach(([methodKey, stats]) => {
    if (methodKey === "original") return;
    const row = document.createElement("tr");
    let methodLabel = methodKey;
    if (isNumeric) {
      row.innerHTML = `
        <td>${methodLabel}</td>
        <td>${toFixedIfNumber(stats.imputed_mean)}</td>
        <td>${toFixedIfNumber(stats.imputed_median)}</td>
        <td>${toFixedIfNumber(stats.imputed_std)}</td>
        <td>${toFixedIfNumber(stats.kde_overlap)}</td>
        <td>${toFixedIfNumber(stats.imputed_skew)}</td>
        <td>${toFixedIfNumber(stats.imputed_kurtosis)}</td>
        <td>${toFixedIfNumber(stats.ks_stat)}</td>
        <td>${toFixedIfNumber(stats.ks_pvalue)}</td>
        <td>${toFixedIfNumber(stats.kl_divergence)}</td>
      `;
    } else {
      const imp = stats.imputed || {};
      row.innerHTML = `
        <td>${methodLabel}</td>
        <td>${imp.mode || "N/A"}</td>
        <td>${imp.unique_count != null ? imp.unique_count : "N/A"}</td>
        <td>${imp.mode_frequency != null ? imp.mode_frequency : "N/A"}</td>
        <td>${imp.unique_percentage != null ? toFixedIfNumber(imp.unique_percentage) + "%" : "N/A"}</td>
      `;
    }
    tableBody.appendChild(row);
  });
}

// ------------------------------------------------
// Apply Imputation for a Column (Temporary)
// ------------------------------------------------
async function applyImputation(col, method, constantVal) {
  try {
    const resp = await fetch("/impute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column: col, method: method, constant_value: constantVal })
    });
    const data = await resp.json();
    if (data.error) { alert("Imputation error: " + data.error); return; }
    if (!columnImputations[col]) columnImputations[col] = {};
    columnImputations[col][method] = data.imputed_distribution;
    const heatmapMatrix = data.updated_matrix;
    if (heatmapMatrix) { updateHeatmap(heatmapMatrix); }
    else { console.warn("Heatmap matrix update missing from backend response!"); }
    appendImputationTraces(col, data.imputed_distribution, method, data.kde_x_imputed, data.kde_y_imputed);
    fetchStatsForColumn(col);
  } catch (err) { console.error("applyImputation error:", err); }
}

// ------------------------------------------------
// Append Imputation Traces to Plots
// ------------------------------------------------
function appendImputationTraces(col, dist, method, kx, ky, superimpose = false) {
  const { histogramId, kdeId, boxId, corrId } = columnPlotData[col];
  const isNumeric = columnDataStore[col] && columnDataStore[col].is_numeric;
  let histTrace;

  if (isNumeric) {
    histTrace = {
      x: dist,
      type: "histogram",
      name: `Imputed(${method})`,
      marker: { color: getRandomColor() },
      opacity: 0.6,
      autobinx: false,
      xbins: {
        start: Math.min(...dist) - 1,
        end: Math.max(...dist) + 1,
        size: 1,
      },
    };
  } else {
    // For categorical columns, compute counts for each unique value.
    const counts = {};
    dist.forEach(val => counts[val] = (counts[val] || 0) + 1);
    const categories = Object.keys(counts);
    const frequencies = Object.values(counts);
    histTrace = {
      x: categories,
      y: frequencies,
      type: "bar",
      name: `Imputed(${method})`,
      marker: { color: getRandomColor() },
      opacity: 0.6,
    };
  }

  Plotly.addTraces(histogramId, [histTrace]);
  Plotly.relayout(histogramId, {
    barmode: isNumeric ? (superimpose ? "overlay" : "group") : undefined,
    legend: {
      orientation: "h",
      x: 0.5,
      y: 1.1,
      xanchor: "center",
      yanchor: "bottom"
    },
    margin: { l: 30, r: 30, t: 30, b: 70 },
    font: { size: 10 }
  });

  // Add KDE and box plot traces only for numeric columns.
  if (isNumeric) {
    if (kdeId && kx && ky) {
      const kdeTrace = {
        x: kx,
        y: ky,
        type: "scatter",
        mode: "lines",
        name: `Imputed(${method}) KDE`,
        line: { color: getRandomColor(), width: 2, dash: "dash" },
      };
      Plotly.addTraces(kdeId, [kdeTrace]);
      Plotly.relayout(kdeId, {
        legend: {
          orientation: "h",
          x: 0.5,
          y: 1.1,
          xanchor: "center",
          yanchor: "bottom"
        },
        margin: { l: 30, r: 30, t: 30, b: 70 },
        font: { size: 10 }
      });
    }

    if (boxId) {
      const boxTrace = {
        x: dist,
        type: "box",
        orientation: "h",
        name: `Imputed(${method}) Box Plot`,
        boxpoints: false,
        jitter: 0.3,
        pointpos: -1.8,
        marker: { color: getRandomColor() },
      };
      Plotly.addTraces(boxId, [boxTrace]);
      Plotly.relayout(boxId, {
        legend: {
          orientation: "h",
          x: 0.5,
          y: 1.1,
          xanchor: "center",
          yanchor: "bottom"
        },
        margin: { l: 10, r: 10, t: 40, b: 10 },
        yaxis: { showticklabels: false },
        font: { size: 10 }
      });
    }

    if (corrId) {
      fetch("/calculate-correlation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ column: col, imputed: dist }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.error) {
            console.error(data.error);
            return;
          }
          const corrTrace = {
            x: data.columns,
            y: data.correlations,
            type: "scatter",
            mode: "lines+markers",
            name: `Imputed(${method}) Correlation`,
            line: { color: getRandomColor(), width: 2, dash: "dot" },
          };
          Plotly.addTraces(corrId, [corrTrace]);
          Plotly.relayout(corrId, {
            legend: {
              orientation: "h",
              x: 0.5,
              y: 1.1,
              xanchor: "center",
              yanchor: "bottom"
            },
            margin: { l: 30, r: 30, t: 30, b: 70 },
            font: { size: 10 }
          });
        })
        .catch((err) =>
          console.error("Error fetching correlation data for imputed column:", err)
        );
    }
  }
}

function showLoading() {
  document.getElementById("loadingOverlay").style.display = "flex";
}

// Function to hide the loading overlay
function hideLoading() {
  document.getElementById("loadingOverlay").style.display = "none";
}

// ------------------------------------------------
// Local KDE Approximation for Imputed Distribution
// ------------------------------------------------
function computeLocalKDEandPlot(col, dist, method) {
  if (!dist || dist.length < 2) {
    const { histogramId } = columnPlotData[col];
    const hTrace = { x: dist, type: "histogram", name: `Imputed(${method})`, marker: { color: getRandomColor() }, opacity: 0.7 };
    Plotly.addTraces(histogramId, [hTrace]);
    return;
  }
  const sorted = [...dist].sort((a, b) => a - b);
  const n = 100;
  const minV = sorted[0];
  const maxV = sorted[sorted.length - 1];
  const step = (maxV - minV) / (n - 1);
  const m = sorted.reduce((a, c) => a + c, 0) / sorted.length;
  const variance = sorted.reduce((a, c) => a + Math.pow(c - m, 2), 0) / (sorted.length - 1);
  const std = Math.sqrt(variance) || 1;
  const bandwidth = 1.06 * std * Math.pow(sorted.length, -0.2);
  const kernel = z => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * z * z);
  const kx = [];
  const ky = [];
  for (let i = 0; i < n; i++) {
    const x = minV + i * step;
    let sVal = 0;
    for (let val of sorted) { sVal += kernel((x - val) / bandwidth); }
    const density = sVal / (sorted.length * bandwidth);
    kx.push(x);
    ky.push(density);
  }
  appendImputationTraces(col, dist, method, kx, ky);
}

// ------------------------------------------------
// Pipeline Functions
// ------------------------------------------------
function addToPipeline(col, method, config) {
  pipeline.push({ column: col, method: method, config: config });
  renderPipeline();
}
function removePipelineStep(idx) {
  pipeline.splice(idx, 1);
  renderPipeline();
}
function renderPipeline() {
  const pc = document.getElementById("pipeline-container");
  pc.innerHTML = "";
  if (pipeline.length === 0) {
    pc.innerHTML = '<p>No pipeline steps added yet.</p>';
    return;
  }
  const container = document.createElement("div");
  container.className = "pipeline-step-container";
  pipeline.forEach((step, i) => {
    const btn = document.createElement("button");
    btn.textContent = `${step.column} → ${step.method}` + (step.config && step.config.value ? ` (Value: ${step.config.value})` : "");
    const delBtn = document.createElement("span");
    delBtn.textContent = "  ✖";
    delBtn.className = "badge";
    delBtn.onclick = () => removePipelineStep(i);
    btn.appendChild(delBtn);
    container.appendChild(btn);
  });
  pc.appendChild(container);
}
function exportPipeline() {
  const pipelineData = JSON.stringify(pipeline, null, 2);
  const blob = new Blob([pipelineData], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "my_pipeline.impvss";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
document.getElementById("pipelineUploadForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const file = document.getElementById("impvss-file").files[0];
  if (!file) { alert("Select a .impvss file!"); return; }
  const reader = new FileReader();
  reader.onload = (evt) => {
    try {
      const pipelineArr = JSON.parse(evt.target.result);
      if (!Array.isArray(pipelineArr)) { alert("Invalid pipeline format!"); return; }
      pipeline = pipelineArr;
      renderPipeline();
      alert("Pipeline imported successfully!");
    } catch (err) { alert("Error parsing pipeline: " + err); }
  };
  reader.readAsText(file);
});
async function applyImportedPipeline() {
  if (!pipeline.length) {
    alert("No pipeline steps to apply");
    return;
  }
  for (let step of pipeline) {
    await applyImputation(step.column, step.method, step.config?.value || null);
    addImputationReference(step.column, step.method, step.config?.value || null);
  }
  alert("All pipeline steps applied!");
}

// ------------------------------------------------
// Utility Functions
// ------------------------------------------------
function toFixedIfNumber(val, decimals = 5) {
  if (typeof val === "number") return val.toFixed(decimals);
  return val == null ? "N/A" : val;
}

  // Event listener for the "Check AI Dependency" button
  checkAIDependencyBtn.addEventListener('click', () => {
    showLoading(); // Show loading overlay

    fetch('/check_ai_readiness', {  // Updated URL
      method: 'GET',  // Changed to GET as per route definition
      headers: {
        'Content-Type': 'application/json'
      },
      // No body needed for GET request
    })
      .then(response => response.json())
      .then(data => {
        hideLoading(); // Hide loading overlay
        if (data.success !== undefined) {  // Adjust based on response structure
          if (!data.ollama_ready) {
            alert(`Error: ${data.error}`);
            return;
          }
          displayAIModules(data.models, data.ollama_ready, data.error);
        } else {
          // If 'success' key is not used in /check_ai_readiness, adjust accordingly
          displayAIModules(data.models, data.ollama_ready, data.error);
        }
      })
      .catch(error => {
        hideLoading(); // Hide loading overlay
        console.error('Error fetching AI dependencies:', error);
        alert('An error occurred while checking AI dependencies.');
      });
  });

  // Function to display AI models
  function displayAIModules(models, ollamaReady, error) {
    if (!ollamaReady) {
      alert(`Error: ${error}`);
      return;
    }

    if (!models || models.length === 0) {
      alert("No Ollama AI models are currently installed.");
      return;
    }

    // Create a formatted string of models
    const modelList = models.join("\n");

    // Display the models in an alert or a better UI component
    alert(`Installed Ollama AI Models:\n\n${modelList}`);

    // Alternatively, display in a modal or a specific div
    /*
    const modalContent = document.getElementById('aiDependencyModalContent');
    modalContent.textContent = modelList;
    document.getElementById('aiDependencyModal').style.display = 'flex';
    */
  }

// Acrylic-inspired soft color palette for rendering graphs.
const acrylicShades = [
  "#8ca6e6", // was "#a8c0ff", now a slightly darker soft blue
  "#a8d8f7", // was "#c7f0ff", now a darker pale cyan
  "#d8db8c", // was "#f2f5a9", now a darker pastel yellow-green
  "#e3bfff", // was "#f9d7ff", now a darker soft lavender
  "#f9aabb", // was "#ffd1dc", now a darker gentle pink
  "#ffcc99", // was "#ffe6b3", now a darker light peach
  "#b6a8d2"  // was "#d1c4e9", now a darker soft mauve
];
let colorIndex = 0;

function getRandomColor() {
  const color = acrylicShades[colorIndex];
  colorIndex = (colorIndex + 1) % acrylicShades.length;
  return color;
}

// ------------------------------------------------
// Imputation Reference Functions
// ------------------------------------------------
function addImputationReference(col, method, config) {
  if (!imputationReferences[col]) {
    imputationReferences[col] = [];
  }
  imputationReferences[col].push({ method: method, config: config });
  renderImputationReference(col);
}

function renderImputationReference(col) {
  const safeCol = makeValidId(col);
  const container = document.querySelector(`#pane-${safeCol} .imputation-reference-container`);

  if (!container) return;

  if (!imputationReferences[col] || imputationReferences[col].length === 0) {
    container.innerHTML = "<em>No imputation reference for this column.</em>";
    return;
  }

  container.innerHTML = "Select to Push in Current Dataset:";

  imputationReferences[col].forEach((ref) => {
    const btn = document.createElement("button");
    btn.textContent = ref.method === "constant"
      ? `${ref.method} (${ref.config || "N/A"})`
      : ref.method;
    btn.onclick = () => pushImputation(col, ref.method, ref.config || null);
    container.appendChild(btn);
  });
}

async function pushImputation(col, method, constantValue = null) {
  try {
    const resp = await fetch("/push-imputation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column: col, method: method, constant_value: constantValue })
    });
    const data = await resp.json();
    if (data.error) {
      alert("Push Imputation Error: " + data.error);
    } else {
      alert(data.message);
      updateHeatmap(data.updated_matrix);
    }
  } catch (err) {
    console.error("pushImputation error:", err);
  }
}
