// app.js — lógica del dashboard AgroPredict.
// Consume la API FastAPI (src/api/main.py) y pinta la Matriz de
// Consistencia + resultados del capítulo 4 + formulario de predicción.

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  loadConsistencyMatrix();
  loadModelComparison();
  loadAblation();
  loadHypothesisReport();
  setupPredictForm();
});

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

// ---------------- Matriz de consistencia ----------------
async function loadConsistencyMatrix() {
  try {
    const m = await fetchJSON("/api/consistency-matrix");
    document.getElementById("pg-text").textContent = m.problema_general;
    document.getElementById("og-text").textContent = m.objetivo_general;
    document.getElementById("hg-text").textContent = m.hipotesis_general;

    const grid = document.getElementById("dimensiones-grid");
    grid.innerHTML = "";
    m.dimensiones.forEach((d, i) => {
      const div = document.createElement("div");
      div.className = `dim-card dim-x${i + 1}`;
      div.innerHTML = `
        <span class="label">${d.codigo} — ${d.nombre}</span>
        <h3>${d.codigo}</h3>
        <p><b>Problema:</b> ${d.problema_especifico}</p>
        <p><b>Objetivo:</b> ${d.objetivo_especifico}</p>
        <p><b>Variables:</b> ${d.variables.join(", ")}</p>
      `;
      grid.appendChild(div);
    });

    const list = document.getElementById("inconsistencias-list");
    list.innerHTML = "";
    (m.inconsistencias_detectadas || []).forEach((txt) => {
      const li = document.createElement("li");
      li.textContent = txt;
      list.appendChild(li);
    });
  } catch (e) {
    console.error("No se pudo cargar la matriz de consistencia:", e);
  }
}

// ---------------- Comparación de modelos ----------------
async function loadModelComparison() {
  try {
    const rows = await fetchJSON("/api/results/model-comparison");
    if (!rows.length) return;

    renderTable("tabla-modelos", rows);

    new Chart(document.getElementById("chart-modelos"), {
      type: "bar",
      data: {
        labels: rows.map((r) => r.Modelo),
        datasets: [{
          label: "R² (TSS, media)",
          data: rows.map((r) => r.R2_TSS_media),
          backgroundColor: ["#1B6B4A", "#C0392B", "#7D3C98", "#1A5276"],
        }],
      },
      options: { scales: { y: { beginAtZero: true, max: 1 } } },
    });
  } catch (e) {
    console.error("No se pudo cargar la comparación de modelos:", e);
  }
}

// ---------------- Ablation study ----------------
async function loadAblation() {
  try {
    const rows = await fetchJSON("/api/results/ablation");
    if (!rows.length) return;

    renderTable("tabla-ablation", rows);

    new Chart(document.getElementById("chart-ablation"), {
      type: "bar",
      data: {
        labels: rows.map((r) => r.Dimension),
        datasets: [{
          label: "R² al excluir la dimensión",
          data: rows.map((r) => r.R2_sin_dim),
          backgroundColor: ["#1A5276", "#1B6B4A", "#C4960A", "#C0392B"],
        }],
      },
      options: { indexAxis: "y", scales: { x: { beginAtZero: true, max: 1 } } },
    });
  } catch (e) {
    console.error("No se pudo cargar el ablation study:", e);
  }
}

// ---------------- Reporte de hipótesis ----------------
async function loadHypothesisReport() {
  try {
    const r = await fetchJSON("/api/results/hypothesis-report");

    document.getElementById("hg-card").innerHTML = `
      <h2>Hipótesis General</h2>
      <p>${r.hipotesis_general.texto}</p>
      <p><b>Evidencia:</b> ${r.hipotesis_general.evidencia}</p>
      <p><b>Estado vs. umbral 0.75 (cap. 4):</b> ${r.hipotesis_general["estado_vs_umbral_0.75_cap4"]}</p>
      <p><b>Estado vs. umbral 0.85 (cap. 3):</b> ${r.hipotesis_general["estado_vs_umbral_0.85_cap3"]}</p>
    `;

    const cont = document.getElementById("he-cards");
    cont.innerHTML = "";
    r.hipotesis_especificas.forEach((h, i) => {
      const div = document.createElement("div");
      div.className = `dim-card dim-x${i + 1}`;
      div.innerHTML = `
        <span class="label">${h.codigo}</span>
        <p>${h.texto}</p>
        <p><b>ΔR²:</b> ${h.diferencia_R2.toFixed(4)} | <b>p-valor:</b> ${h.p_valor.toFixed(4)}</p>
        <p><b>Significativa (Wilcoxon):</b> ${h.significativa_wilcoxon}</p>
        <p><b>Importancia Gini de la dimensión:</b> ${(h.importancia_gini_dimension * 100).toFixed(2)}%</p>
      `;
      cont.appendChild(div);
    });
  } catch (e) {
    console.error("No se pudo cargar el reporte de hipótesis:", e);
  }
}

// ---------------- Predicción ----------------
function setupPredictForm() {
  const form = document.getElementById("predict-form");
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());
    Object.keys(data).forEach((k) => (data[k] = Number(data[k])));

    const resultBox = document.getElementById("predict-result");
    resultBox.textContent = "Calculando...";
    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || "Error");
      resultBox.innerHTML = `<h3>Precio predicho: S/ ${json.precio_predicho_S_por_kg} / kg</h3>`;
    } catch (e) {
      resultBox.textContent = `Error: ${e.message}`;
    }
  });
}

// ---------------- Utilidad ----------------
function renderTable(elId, rows) {
  const table = document.getElementById(elId);
  const cols = Object.keys(rows[0]);
  table.innerHTML = `
    <thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${cols.map((c) => {
      const v = r[c];
      return `<td>${typeof v === "number" ? v.toFixed(4) : v}</td>`;
    }).join("")}</tr>`).join("")}</tbody>
  `;
}
