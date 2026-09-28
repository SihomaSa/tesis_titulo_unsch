// app.js — lógica del dashboard AgroPredict.
// Consume la API FastAPI (src/api/main.py) y pinta la Matriz de
// Consistencia + resultados del capítulo 4 + formulario de predicción.
const API_BASE_URL = String(window.API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

// Umbrales de R² (espejo de src/config.py)
const UMBRAL_OFICIAL = 0.85;
const UMBRAL_REFERENCIA = 0.75;
const N_SPLITS_TSS = 6;

// Colores por dimensión
const DIM_COLORS = { X1: "#1A5276", X2: "#1B6B4A", X3: "#C4960A", X4: "#C0392B" };
const DIM_NOMBRES = {
  X1_Temporal: "X1 Temporal",
  X2_Productiva: "X2 Productiva",
  X3_Geografica: "X3 Geográfica",
  X4_Economica_Historica: "X4 Económica histórica",
};

const COLUMN_LABELS = {
  Modelo: "Modelo",
  R2_TSS_media: "R² medio (TSS)",
  R2_TSS_std: "Desv. estándar R²",
  MAE_TSS: "MAE",
  RMSE_TSS: "RMSE",
  Dimension: "Dimensión",
  R2_sin_dim: "R² sin la dimensión",
  R2_completo: "R² modelo completo",
  Diferencia_R2: "ΔR²",
  p_valor: "p-valor (Wilcoxon)",
  Significativa_estadistica: "Significativa",
};

// ============================================================
// MAPEO nombre → código del LabelEncoder
// Orden alfabético (así los asigna sklearn.preprocessing.LabelEncoder)
// Si tu backend imprimió otro orden, ajusta los números aquí.
// ============================================================

// Provincias ordenadas alfabéticamente (dataset_agricola.csv):
// AYABACA, HUANCABAMBA, MORROPON, PAITA, PIURA, SECHURA, SULLANA, TALARA
const PROVINCIA_A_CODIGO = {
  "AYABACA":     0,
  "HUANCABAMBA": 1,
  "MORROPON":    2,
  "PAITA":       3,
  "PIURA":       4,
  "SECHURA":     5,
  "SULLANA":     6,
  "TALARA":      7,
};

// Distritos ordenados alfabéticamente (todos los que aparecen en el dataset).
// ⚠️ Este es el orden GLOBAL (no por provincia) que genera LabelEncoder.
// Si tu encoder fue entrenado con un subconjunto distinto, ajusta.
const DISTRITO_A_CODIGO = {
  "AMOTAPE":                    0,
  "ARENAL":                     1,
  "AYABACA":                    2,
  "BELLAVISTA DE LA UNION":     3,
  "BERNAL":                     4,
  "BUENOS AIRES":               5,
  "CANCHAQUE":                  6,
  "CASTILLA":                   7,
  "CATACAOS":                   8,
  "CHALACO":                    9,
  "CHULUCANAS":                10,
  "COLAN":                     11,
  "CRISTO NOS VALGA":          12,
  "CURA MORI":                 13,
  "EL ALTO":                   14,
  "EL CARMEN DE LA FRONTERA":  15,
  "EL TALLAN":                 16,
  "FRIAS":                     17,
  "HUANCABAMBA":               18,
  "HUARMACA":                  19,
  "IGNACIO ESCUDERO":          20,
  "JILILI":                    21,
  "LA ARENA":                  22,
  "LA BREA":                   23,
  "LA HUACA":                  24,
  "LA MATANZA":                25,
  "LA UNION":                  26,
  "LAGUNAS":                   27,
  "LALAQUIZ":                  28,
  "LAS LOMAS":                 29,
  "LOBITOS":                   30,
  "LOS ORGANOS":               31,
  "MANCORA":                   32,
  "MARCAVELICA":               33,
  "MIGUEL CHECA":              34,
  "MONTERO":                   35,
  "MORROPON":                  36,
  "PACAIPAMPA":                37,
  "PAIMAS":                    38,
  "PAITA":                     39,
  "PARIÑAS":                   40,
  "PIURA":                     41,
  "QUERECOTILLO":              42,
  "RINCONADA LLICUAR":         43,
  "SALITRAL":                  44,
  "SAN JUAN DE BIGOTE":        45,
  "SAN MIGUEL DE EL FAIQUE":   46,
  "SANTA CATALINA DE MOSSA":   47,
  "SANTO DOMINGO":             48,
  "SAPILLICA":                 49,
  "SECHURA":                   50,
  "SICCHEZ":                   51,
  "SONDOR":                    52,
  "SONDORILLO":                53,
  "SULLANA":                   54,
  "SUYO":                      55,
  "TAMARINDO":                 56,
  "TAMBO GRANDE":              57,
  "VICE":                      58,
  "VICHAYAL":                  59,
  "YAMANGO":                   60,
};

// Distritos disponibles por provincia (para el filtro dinámico del <select>)
const DISTRITOS_POR_PROVINCIA = {
  PIURA:       ["PIURA", "CASTILLA", "CATACAOS", "CURA MORI", "EL TALLAN", "LA ARENA", "LA UNION", "LAS LOMAS", "TAMBO GRANDE"],
  AYABACA:     ["AYABACA", "FRIAS", "JILILI", "LAGUNAS", "MONTERO", "PACAIPAMPA", "PAIMAS", "SAPILLICA", "SICCHEZ", "SUYO"],
  HUANCABAMBA: ["HUANCABAMBA", "CANCHAQUE", "EL CARMEN DE LA FRONTERA", "HUARMACA", "LALAQUIZ", "SAN MIGUEL DE EL FAIQUE", "SONDOR", "SONDORILLO"],
  MORROPON:    ["CHULUCANAS", "BUENOS AIRES", "CHALACO", "LA MATANZA", "MORROPON", "SALITRAL", "SAN JUAN DE BIGOTE", "SANTA CATALINA DE MOSSA", "SANTO DOMINGO", "YAMANGO"],
  PAITA:       ["AMOTAPE", "ARENAL", "COLAN", "LA HUACA", "TAMARINDO", "VICHAYAL"],
  SULLANA:     ["SULLANA", "IGNACIO ESCUDERO", "MARCAVELICA", "MIGUEL CHECA", "QUERECOTILLO", "SALITRAL"],
  SECHURA:     ["SECHURA", "BELLAVISTA DE LA UNION", "BERNAL", "CRISTO NOS VALGA", "RINCONADA LLICUAR", "VICE"],
  TALARA:      ["PARIÑAS", "EL ALTO", "LA BREA", "LOBITOS", "LOS ORGANOS", "MANCORA"],
};

const charts = {};

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupFigures();
  setupChartDefaults();
  loadConsistencyMatrix();
  loadModelComparison();
  loadAblation();
  loadHypothesisReport();
  setupPredictForm();
  setupFiltroProvinciaDistrito();
});

// ---------------- Utilidades ----------------
const $ = (id) => document.getElementById(id);
const isNum = (v) => typeof v === "number" && Number.isFinite(v);

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmt(v, dec = 4) {
  const n = typeof v === "string" && v.trim() !== "" ? Number(v) : v;
  return isNum(n) ? n.toFixed(dec) : "—";
}

function fmtSigned(v, dec = 4) {
  if (!isNum(v)) return "—";
  const s = Math.abs(v).toFixed(dec);
  if (Number(s) === 0) return s;
  return (v < 0 ? "−" : "+") + s;
}

function fmtPct(v, dec = 2) {
  return isNum(v) ? `${(v * 100).toFixed(dec)}%` : "—";
}

function esSi(v) {
  return v === true || /^(s[ií]|true|yes|1)$/i.test(String(v ?? "").trim());
}

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function showFetchError(elId, mensaje) {
  const el = $(elId);
  if (el) el.textContent = mensaje;
}

function buscarValorPorPatron(obj, patron) {
  if (!obj) return undefined;
  const clave = Object.keys(obj).find((k) => patron.test(k));
  return clave !== undefined ? obj[clave] : undefined;
}

function normalizarEstado(v) {
  if (v === undefined || v === null || v === "") return null;
  const s = String(v).trim().toUpperCase();
  if (s.startsWith("ACEPT")) return "ACEPTADA";
  if (s.startsWith("RECHAZ")) return "RECHAZADA";
  return null;
}

function dimNumero(codigo, i) {
  const m = String(codigo ?? "").match(/[1-4]/);
  return m ? m[0] : String(Math.min(i + 1, 4));
}

function titleCase(str) {
  return String(str)
    .toLowerCase()
    .replace(/(^|\s|\(|\/)([a-záéíóúñ])/g, (_, sep, ch) => sep + ch.toUpperCase());
}

// ---------------- Tabs ----------------
function setupTabs() {
  const btns = Array.from(document.querySelectorAll(".tab-btn"));

  function activate(id, { focus = false, updateHash = true } = {}) {
    const btn = btns.find((b) => b.dataset.tab === id) || btns[0];
    btns.forEach((b) => {
      const on = b === btn;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
    });
    document.querySelectorAll(".tab-content").forEach((c) => {
      const on = c.id === btn.dataset.tab;
      c.classList.toggle("active", on);
      c.hidden = !on;
    });
    if (focus) btn.focus();
    if (updateHash) history.replaceState(null, "", `#${btn.dataset.tab}`);
    Object.values(charts).forEach((c) => c.resize());
  }

  btns.forEach((btn, i) => {
    btn.addEventListener("click", () => activate(btn.dataset.tab));
    btn.addEventListener("keydown", (ev) => {
      const last = btns.length - 1;
      let next = null;
      if (ev.key === "ArrowRight") next = i === last ? 0 : i + 1;
      else if (ev.key === "ArrowLeft") next = i === 0 ? last : i - 1;
      else if (ev.key === "Home") next = 0;
      else if (ev.key === "End") next = last;
      if (next !== null) {
        ev.preventDefault();
        activate(btns[next].dataset.tab, { focus: true });
      }
    });
  });

  const inicial = location.hash.replace("#", "");
  if (btns.some((b) => b.dataset.tab === inicial)) activate(inicial, { updateHash: false });
}

// ---------------- Figuras del pipeline ----------------
function setupFigures() {
  document.querySelectorAll(".figures-grid figure").forEach((fig) => {
    const img = fig.querySelector("img");
    if (!img) return;
    const marcarFaltante = () => {
      if (fig.classList.contains("is-missing")) return;
      fig.classList.add("is-missing");
      const aviso = document.createElement("div");
      aviso.className = "fig-missing";
      aviso.innerHTML =
        `No se encontró la imagen <code>${esc(img.getAttribute("src"))}</code>.<br />` +
        "Descárgala desde MLflow (Artifacts) y guárdala en esa ruta.";
      fig.insertBefore(aviso, img);
    };
    img.addEventListener("error", marcarFaltante);
    if (img.complete && img.naturalWidth === 0 && img.getAttribute("src")) marcarFaltante();
  });
}

// ---------------- Chart.js ----------------
function setupChartDefaults() {
  if (typeof Chart === "undefined") return;
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 12;
  Chart.defaults.color = "#57655C";
}

const referenceLine = {
  id: "referenceLine",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !isNum(opts.value)) return;
    const { ctx, chartArea, scales } = chart;
    const x = scales.x.getPixelForValue(opts.value);
    if (x < chartArea.left || x > chartArea.right) return;
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#1C2621";
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top - 6);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#1C2621";
    ctx.font = `600 11px ${Chart.defaults.font.family}`;
    ctx.textBaseline = "bottom";
    ctx.textAlign = opts.align || "center";
    const dx = opts.align === "right" ? -6 : opts.align === "left" ? 6 : 0;
    ctx.fillText(opts.label || "", x + dx, chartArea.top - 8);
    ctx.restore();
  },
};

const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    const ds = chart.data.datasets[0];
    ctx.save();
    ctx.font = `600 12px ${Chart.defaults.font.family}`;
    ctx.fillStyle = "#1C2621";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    chart.getDatasetMeta(0).data.forEach((bar, i) => {
      ctx.fillText(fmt(ds.data[i]), bar.x + 8, bar.y);
    });
    ctx.restore();
  },
};

function opcionesBarrasHorizontales(refOpts) {
  return {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    layout: { padding: { top: 26, right: 52 } },
    scales: {
      x: { min: 0, max: 1, grid: { color: "#E9ECE8" }, border: { display: false }, ticks: { stepSize: 0.2 } },
      y: { grid: { display: false }, ticks: { color: "#1C2621", font: { size: 12.5 } } },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${fmt(ctx.parsed.x)}` },
      },
      referenceLine: refOpts,
    },
  };
}

function colorModelo(nombre, i) {
  const n = String(nombre).toLowerCase();
  if (n.includes("random")) return "#1B6B4A";
  if (n.includes("xgb")) return "#C0392B";
  if (n.includes("lightgbm") || n.includes("lgbm")) return "#7D3C98";
  if (n.includes("lineal") || n.includes("linear")) return "#1A5276";
  return ["#1B6B4A", "#C0392B", "#7D3C98", "#1A5276", "#C4960A"][i % 5];
}

function mostrarErrorGrafico(wrap, mensaje) {
  wrap.classList.remove("is-loading");
  wrap.innerHTML = `<p class="result-error">${esc(mensaje)}</p>`;
}

// ---------------- Matriz de consistencia ----------------
async function loadConsistencyMatrix() {
  try {
    const m = await fetchJSON("/api/consistency-matrix");
    showFetchError("pg-text", m.problema_general ?? "—");
    showFetchError("og-text", m.objetivo_general ?? "—");
    showFetchError("hg-text", m.hipotesis_general ?? "—");

    const dims = Array.isArray(m.dimensiones) ? m.dimensiones : [];
    $("dimensiones-grid").innerHTML = dims.map((d, i) => {
      const n = dimNumero(d.codigo, i);
      const vars = (d.variables || []).map((v) => `<li>${esc(v)}</li>`).join("");
      return `
        <article class="dim-card dim-x${n}">
          <header class="dim-head">
            <h3>${esc(d.nombre)}</h3>
            <span class="dim-index">${esc(d.codigo)}</span>
          </header>
          <dl class="dim-fields">
            <div><dt>Problema</dt><dd>${esc(d.problema_especifico)}</dd></div>
            <div><dt>Objetivo</dt><dd>${esc(d.objetivo_especifico)}</dd></div>
          </dl>
          <div class="dim-vars">
            <span class="dim-vars-title">Variables</span>
            <ul class="chips">${vars}</ul>
          </div>
        </article>`;
    }).join("");
  } catch (e) {
    console.error("No se pudo cargar la matriz de consistencia:", e);
    ["pg-text", "og-text", "hg-text"].forEach((id) =>
      showFetchError(id, "No se pudo conectar con la API. Verifica que el backend esté activo.")
    );
    $("dimensiones-grid").innerHTML = "";
  }
}

// ---------------- Comparación de modelos ----------------
async function loadModelComparison() {
  const wrap = $("chart-modelos-wrap");
  try {
    const rows = await fetchJSON("/api/results/model-comparison");
    wrap.classList.remove("is-loading");
    if (!rows.length) {
      mostrarErrorGrafico(wrap, "Aún no hay resultados. Corre main.py en el backend.");
      return;
    }

    const mejor = Math.max(...rows.map((r) => Number(r.R2_TSS_media)));
    renderTable("tabla-modelos", rows, {
      mejorFila: (r) => Number(r.R2_TSS_media) === mejor,
      celda: (col, r, i) =>
        col === "Modelo"
          ? `<span class="dot" style="background:${colorModelo(r.Modelo, i)}"></span>${esc(r.Modelo)}` +
            (Number(r.R2_TSS_media) === mejor ? '<span class="badge best">Mayor R²</span>' : "")
          : null,
    });

    if (typeof Chart === "undefined") {
      mostrarErrorGrafico(wrap, "No se pudo cargar Chart.js (revisa tu conexión a internet).");
      return;
    }
    charts.modelos = new Chart($("chart-modelos"), {
      type: "bar",
      data: {
        labels: rows.map((r) => r.Modelo),
        datasets: [{
          label: "R² (TSS, media)",
          data: rows.map((r) => Number(r.R2_TSS_media)),
          backgroundColor: rows.map((r, i) => colorModelo(r.Modelo, i)),
          borderRadius: 4,
          maxBarThickness: 34,
        }],
      },
      options: opcionesBarrasHorizontales({ value: UMBRAL_OFICIAL, label: `Umbral ${UMBRAL_OFICIAL}`, align: "center" }),
      plugins: [referenceLine, valueLabels],
    });
  } catch (e) {
    console.error("No se pudo cargar la comparación de modelos:", e);
    mostrarErrorGrafico(wrap, "No se pudo cargar el gráfico. Verifica la conexión con la API.");
  }
}

// ---------------- Ablation study ----------------
async function loadAblation() {
  const wrap = $("chart-ablation-wrap");
  try {
    const rows = await fetchJSON("/api/results/ablation");
    wrap.classList.remove("is-loading");
    if (!rows.length) {
      mostrarErrorGrafico(wrap, "Aún no hay resultados. Corre main.py en el backend.");
      return;
    }

    renderTable("tabla-ablation", rows, {
      celda: (col, r) => {
        if (col === "Dimension") {
          const c = DIM_COLORS[String(r.Dimension).slice(0, 2)] || "#57655C";
          return `<span class="dot" style="background:${c}"></span>${esc(DIM_NOMBRES[r.Dimension] || String(r.Dimension).replace(/_/g, " "))}`;
        }
        if (col === "Significativa_estadistica") {
          const si = esSi(r[col]);
          return `<span class="badge ${si ? "yes" : "no"}">${si ? "Sí" : "No"}</span>`;
        }
        if (col === "Diferencia_R2") {
          const v = Number(r[col]);
          const cls = !isNum(v) || Math.abs(v) < 0.00005 ? "" : v < 0 ? "neg" : "pos";
          return `<span class="${cls}">${fmtSigned(v)}</span>`;
        }
        return null;
      },
    });

    if (typeof Chart === "undefined") {
      mostrarErrorGrafico(wrap, "No se pudo cargar Chart.js (revisa tu conexión a internet).");
      return;
    }
    const completo = Number(rows[0].R2_completo);
    charts.ablation = new Chart($("chart-ablation"), {
      type: "bar",
      data: {
        labels: rows.map((r) => DIM_NOMBRES[r.Dimension] || String(r.Dimension).replace(/_/g, " ")),
        datasets: [{
          label: "R² al excluir la dimensión",
          data: rows.map((r) => Number(r.R2_sin_dim)),
          backgroundColor: rows.map((r) => DIM_COLORS[String(r.Dimension).slice(0, 2)] || "#57655C"),
          borderRadius: 4,
          maxBarThickness: 34,
        }],
      },
      options: opcionesBarrasHorizontales({ value: completo, label: `Modelo completo ${fmt(completo)}`, align: "right" }),
      plugins: [referenceLine, valueLabels],
    });
  } catch (e) {
    console.error("No se pudo cargar el ablation study:", e);
    mostrarErrorGrafico(wrap, "No se pudo cargar el gráfico. Verifica la conexión con la API.");
  }
}

// ---------------- Reporte de hipótesis ----------------
function extraerR2(modeloFinal, evidencia) {
  const directo = Number(modeloFinal?.r2_final);
  if (isNum(directo)) return directo;
  const m = String(evidencia ?? "").match(/=\s*(\d+(?:[.,]\d+)?)/);
  return m ? parseFloat(m[1].replace(",", ".")) : null;
}

function chipEstado(estado) {
  if (estado === "ACEPTADA") return '<span class="chip ok">Aceptada</span>';
  if (estado === "RECHAZADA") return '<span class="chip bad">Rechazada</span>';
  return '<span class="chip na">Sin dato</span>';
}

function renderHipotesisGeneral(r) {
  const hg = r.hipotesis_general || {};
  const mf = r.modelo_final || {};
  const r2 = extraerR2(mf, hg.evidencia);

  const calc = (u) => (isNum(r2) ? (r2 >= u ? "ACEPTADA" : "RECHAZADA") : null);
  const estadoOficial =
    normalizarEstado(buscarValorPorPatron(hg, /oficial/i)) ??
    normalizarEstado(buscarValorPorPatron(hg, /umbral.*0[.,]?85/i)) ??
    calc(UMBRAL_OFICIAL);
  const estadoRef =
    normalizarEstado(buscarValorPorPatron(hg, /umbral.*0[.,]?75/i)) ?? calc(UMBRAL_REFERENCIA);

  const ancho = isNum(r2) ? Math.max(0, Math.min(1, r2)) * 100 : 0;
  const nivel = !isNum(r2) ? "" : r2 >= UMBRAL_OFICIAL ? "" : r2 >= UMBRAL_REFERENCIA ? "mid" : "low";

  const track = isNum(r2) ? `
    <div class="track-wrap">
      <div class="track" role="img" aria-label="R² final ${fmt(r2)} sobre una escala de 0 a 1">
        <div class="track-fill ${nivel}" style="width:${ancho}%"></div>
        <div class="tick ref" style="left:${UMBRAL_REFERENCIA * 100}%"><span>${UMBRAL_REFERENCIA.toFixed(2)}</span></div>
        <div class="tick off" style="left:${UMBRAL_OFICIAL * 100}%"><span>${UMBRAL_OFICIAL.toFixed(2)}</span></div>
      </div>
    </div>` : "";

  const meta = [];
  if (r.modelo_seleccionado) meta.push(["Modelo final", esc(r.modelo_seleccionado)]);
  if (isNum(Number(r.n_registros_validos))) meta.push(["Registros válidos", Number(r.n_registros_validos).toLocaleString("es-PE")]);
  if (isNum(Number(mf.mae_final))) meta.push(["MAE", fmt(Number(mf.mae_final))]);
  if (isNum(Number(mf.rmse_final))) meta.push(["RMSE", fmt(Number(mf.rmse_final))]);
  if (isNum(Number(mf.mape_final))) meta.push(["MAPE", `${fmt(Number(mf.mape_final), 1)}%`]);

  $("hg-card").innerHTML = `
    <article class="verdict">
      <p class="verdict-claim">${esc(hg.texto ?? "—")}</p>
      <div class="verdict-main">
        <div class="score">
          <span class="score-num">${fmt(r2)}</span>
          <span class="score-label">R² final (hold-out)</span>
        </div>
        ${track}
      </div>
      <ul class="verdict-status">
        <li>${chipEstado(estadoOficial)} Umbral oficial ${UMBRAL_OFICIAL.toFixed(2)} (cap. 3)</li>
        <li>${chipEstado(estadoRef)} Referencia ${UMBRAL_REFERENCIA.toFixed(2)} (cap. 4)</li>
      </ul>
      ${meta.length ? `<dl class="verdict-meta">${meta.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>` : ""}
      ${hg.evidencia ? `<p class="verdict-evidence"><b>Evidencia:</b> ${esc(hg.evidencia)}</p>` : ""}
    </article>`;
}

function renderHipotesisEspecificas(r) {
  const lista = Array.isArray(r.hipotesis_especificas) ? r.hipotesis_especificas : [];
  $("he-cards").innerHTML = lista.map((h, i) => {
    const n = dimNumero(h.codigo, i);
    const sig = esSi(h.significativa_wilcoxon);
    const gini = Number(h.importancia_gini_dimension);
    const ancho = isNum(gini) ? Math.max(0, Math.min(1, gini)) * 100 : 0;
    return `
      <article class="dim-card dim-x${n}">
        <header class="dim-head">
          <span class="dim-index">${esc(h.codigo)}</span>
          <span class="badge ${sig ? "sig-yes" : "sig-no"}">${sig ? "Significativa" : "No significativa"}</span>
        </header>
        <p class="hyp-text">${esc(h.texto)}</p>
        <dl class="dim-stats">
          <div><dt>ΔR²</dt><dd>${fmtSigned(Number(h.diferencia_R2))}</dd></div>
          <div><dt>p-valor</dt><dd>${fmt(Number(h.p_valor))}</dd></div>
          <div><dt>Import. Gini</dt><dd>${fmtPct(gini)}</dd></div>
        </dl>
        <div class="share" aria-hidden="true">
          <div class="share-track"><div class="share-fill" style="width:${ancho}%"></div></div>
          <span class="share-label">Peso de la dimensión en el modelo</span>
        </div>
      </article>`;
  }).join("");

  const pMin = 2 / Math.pow(2, N_SPLITS_TSS);
  $("he-note").textContent = lista.length
    ? `Prueba de Wilcoxon sobre ${N_SPLITS_TSS} particiones temporales: el menor p-valor posible es ${pMin.toFixed(4)}.`
    : "";
}

async function loadHypothesisReport() {
  try {
    const r = await fetchJSON("/api/results/hypothesis-report");
    renderHipotesisGeneral(r);
    renderHipotesisEspecificas(r);
  } catch (e) {
    console.error("No se pudo cargar el reporte de hipótesis:", e);
    $("hg-card").innerHTML =
      '<p class="result-error">No se pudo cargar el reporte de hipótesis. Verifica la conexión con la API.</p>';
    $("he-cards").innerHTML = "";
  }
}

// ---------------- Predicción ----------------
function mensajeDeError(json, status) {
  const d = json && json.detail;
  if (Array.isArray(d)) {
    return d.map((x) => `${(x.loc || []).slice(-1)[0] ?? "campo"}: ${x.msg}`).join("; ");
  }
  if (typeof d === "string" && d) return d;
  return `Error del servidor (${status})`;
}

// Filtro dinámico Provincia → Distrito
function setupFiltroProvinciaDistrito() {
  const selProv = $("input-provincia");
  const selDist = $("input-distrito");
  if (!selProv || !selDist) return;

  function repoblarDistritos(provincia, distritoSeleccionado = null) {
    const distritos = DISTRITOS_POR_PROVINCIA[provincia] || [];
    selDist.innerHTML = distritos
      .map((d) => {
        const sel = distritoSeleccionado && d === distritoSeleccionado ? " selected" : "";
        return `<option value="${esc(d)}"${sel}>${esc(titleCase(d))}</option>`;
      })
      .join("");
  }

  selProv.addEventListener("change", () => {
    repoblarDistritos(selProv.value);
  });

  repoblarDistritos(selProv.value, selDist.value || "PIURA");
}

function setupPredictForm() {
  const form = $("predict-form");
  const siembra = $("input-siembra");
  const cosecha = $("input-cosecha");
  const noteCosecha = $("note-cosecha");
  const submitBtn = $("predict-submit");
  const resultBox = $("predict-result");
  const vacio = resultBox.innerHTML;

  function validarCosecha() {
    const s = parseFloat(siembra.value) || 0;
    const c = parseFloat(cosecha.value) || 0;
    const excede = c > s;
    noteCosecha.textContent = excede ? "No puede superar la superficie sembrada" : "";
    noteCosecha.classList.toggle("warn", excede);
    cosecha.classList.toggle("is-invalid", excede);
    return !excede;
  }
  siembra.addEventListener("input", validarCosecha);
  cosecha.addEventListener("input", validarCosecha);

  form.addEventListener("reset", () => {
    setTimeout(() => {
      validarCosecha();
      resultBox.innerHTML = vacio;
      const selProv = $("input-provincia");
      const selDist = $("input-distrito");
      if (selProv && selDist) {
        const distritos = DISTRITOS_POR_PROVINCIA[selProv.value] || [];
        selDist.innerHTML = distritos
          .map((d) => `<option value="${esc(d)}">${esc(titleCase(d))}</option>`)
          .join("");
      }
    }, 0);
  });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();

    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }
    if (!validarCosecha()) {
      cosecha.focus();
      return;
    }

    const raw = Object.fromEntries(new FormData(form).entries());

    // Traducir nombre → código del LabelEncoder
    const provinciaEnc = PROVINCIA_A_CODIGO[raw.PROVINCIA];
    const distritoEnc  = DISTRITO_A_CODIGO[raw.DISTRITO];
    const cultivoEnc   = Number(raw.CULTIVO);

    // Validaciones defensivas
    if (provinciaEnc === undefined) {
      resultBox.innerHTML = `<p class="result-error" role="alert"><b>Provincia no reconocida:</b> "${esc(raw.PROVINCIA)}". Revisa el mapeo del LabelEncoder en <code>app.js</code>.</p>`;
      return;
    }
    if (distritoEnc === undefined) {
      resultBox.innerHTML = `<p class="result-error" role="alert"><b>Distrito no reconocido:</b> "${esc(raw.DISTRITO)}". Revisa el mapeo del LabelEncoder en <code>app.js</code>.</p>`;
      return;
    }
    if (!Number.isFinite(cultivoEnc)) {
      resultBox.innerHTML = `<p class="result-error" role="alert"><b>Cultivo no reconocido:</b> "${esc(raw.CULTIVO)}".</p>`;
      return;
    }

    // Payload con los nombres que el backend FastAPI espera
    const data = {
      ANO:            Number(raw.ANO),
      MES:            Number(raw.MES),
      SIEMBRA:        Number(raw.SIEMBRA),
      COSECHA:        Number(raw.COSECHA),
      PRODUCCION:     Number(raw.PRODUCCION),
      PRECIO_LAG1:    Number(raw.PRECIO_LAG1),
      MEDIA_MOVIL_3:  Number(raw.MEDIA_MOVIL_3),
      PROVINCIA_enc:  provinciaEnc,
      DISTRITO_enc:   distritoEnc,
      CULTIVO_enc:    cultivoEnc,
    };

    submitBtn.disabled = true;
    submitBtn.textContent = "Calculando…";
    resultBox.innerHTML = '<div class="result-loading"><p>Consultando el modelo…</p></div>';

    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(mensajeDeError(json, res.status));

      const precio = Number(json.precio_predicho_S_por_kg);
      if (!isNum(precio)) throw new Error("La API no devolvió un precio válido.");

      const anterior = data.PRECIO_LAG1;
      const deltaPct = anterior > 0 ? ((precio - anterior) / anterior) * 100 : null;
      const deltaHtml = deltaPct === null ? "" : `
        <span class="result-delta ${deltaPct >= 0 ? "up" : "down"}">
          ${deltaPct >= 0 ? "▲" : "▼"} ${Math.abs(deltaPct).toFixed(1)}% frente al mes anterior
        </span>`;

      resultBox.innerHTML = `
        <div class="result-card">
          <p class="result-label">Precio estimado</p>
          <p class="result-price">S/ ${precio.toFixed(3)}<span class="unit"> por kg</span></p>
          ${deltaHtml}
          <dl class="result-compare">
            <div><dt>Mes anterior</dt><dd>S/ ${fmt(anterior, 3)}</dd></div>
            <div><dt>Media móvil 3 meses</dt><dd>S/ ${fmt(data.MEDIA_MOVIL_3, 3)}</dd></div>
          </dl>
        </div>`;
    } catch (e) {
      const sinConexion = e instanceof TypeError;
      const msg = sinConexion
        ? `No se pudo conectar con la API en ${API_BASE_URL}. Verifica que el backend esté activo.`
        : e.message;
      resultBox.innerHTML = `<p class="result-error" role="alert"><b>No se pudo calcular el precio.</b><br />${esc(msg)}</p>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Predecir precio";
    }
  });
}

// ---------------- Tablas ----------------
function renderTable(elId, rows, opciones = {}) {
  const table = $(elId);
  const cols = Object.keys(rows[0]);
  const esNumerica = (c) => isNum(Number(rows[0][c])) && typeof rows[0][c] === "number";
  const head = cols.map((c) =>
    `<th class="${esNumerica(c) ? "num" : ""}">${esc(COLUMN_LABELS[c] || c.replace(/_/g, " "))}</th>`
  ).join("");
  const body = rows.map((r, i) => {
    const tds = cols.map((c) => {
      const custom = opciones.celda ? opciones.celda(c, r, i) : null;
      const contenido = custom !== null && custom !== undefined
        ? custom
        : esNumerica(c) ? fmt(r[c]) : esc(r[c]);
      return `<td class="${esNumerica(c) ? "num" : ""}">${contenido}</td>`;
    }).join("");
    const mejor = opciones.mejorFila && opciones.mejorFila(r) ? "is-best" : "";
    return `<tr class="${mejor}">${tds}</tr>`;
  }).join("");
  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
}