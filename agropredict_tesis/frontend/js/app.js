// ---------------- Reporte de hipótesis ----------------
const num = (v) => (v === null || v === undefined || v === "" ? null : (Number.isFinite(Number(v)) ? Number(v) : null));

function estadoHip(v) {
  if (v === undefined || v === null || v === "") return null;
  const s = String(v).trim().toUpperCase();
  if (s.includes("CONCLUY")) return "NO CONCLUYENTE";
  if (s.includes("PARCIAL")) return "PARCIALMENTE RESPALDADA";
  if (s.startsWith("NO ") || s.startsWith("RECHAZ")) return "NO RESPALDADA";
  if (s.startsWith("RESPALD") || s.startsWith("ACEPT")) return "RESPALDADA";
  return null;
}

function chipEstado(estado) {
  const e = estadoHip(estado);
  if (e === "RESPALDADA") return '<span class="chip ok">Respaldada</span>';
  if (e === "PARCIALMENTE RESPALDADA") return '<span class="chip warn">Parcialmente respaldada</span>';
  if (e === "NO RESPALDADA") return '<span class="chip bad">No respaldada</span>';
  if (e === "NO CONCLUYENTE") return '<span class="chip na">No concluyente</span>';
  return '<span class="chip na">Sin dato</span>';
}

function renderHipotesisGeneral(r) {
  const hg = r.hipotesis_general || {};
  const mf = r.modelo_final || {};
  const a = hg.hg_a || {};
  const b = hg.hg_b || {};
  const crit = num(a.criterio_referencia) ?? UMBRAL_OFICIAL;
  const r2 = num(mf.r2_final);
  const ic = Array.isArray(mf.r2_ic95) ? mf.r2_ic95.map(num) : null;
  const icOk = ic && ic[0] !== null && ic[1] !== null;
  const icTxt = icOk ? ` · IC95 [${fmt(ic[0])}; ${fmt(ic[1])}]` : "";

  const ancho = r2 !== null ? Math.max(0, Math.min(1, r2)) * 100 : 0;
  const nivel = r2 !== null && r2 < crit ? "low" : "";
  const banda = icOk
    ? `<div style="position:absolute;top:0;bottom:0;left:${Math.max(0, ic[0]) * 100}%;width:${Math.max(0, (ic[1] - ic[0])) * 100}%;background:rgba(28,38,33,.28);border-radius:3px" title="IC95 del R²"></div>`
    : "";

  const track = r2 !== null ? `
    <div class="track-wrap">
      <div class="track" role="img" aria-label="R² final ${fmt(r2)} sobre una escala de 0 a 1">
        <div class="track-fill ${nivel}" style="width:${ancho}%"></div>
        ${banda}
        <div class="tick off" style="left:${crit * 100}%"><span>${crit.toFixed(2)}</span></div>
      </div>
    </div>` : "";

  const meta = [];
  if (r.modelo_seleccionado) meta.push(["Modelo final", esc(r.modelo_seleccionado)]);
  if (num(r.n_registros_validos) !== null) meta.push(["Registros válidos", num(r.n_registros_validos).toLocaleString("es-PE")]);
  if (num(r.horizonte_meses) !== null) meta.push(["Horizonte", `${num(r.horizonte_meses)} meses`]);
  if (num(mf.n_test) !== null) meta.push(["n prueba", num(mf.n_test).toLocaleString("es-PE")]);
  if (num(mf.mae_final) !== null) meta.push(["MAE", fmt(num(mf.mae_final))]);
  if (num(mf.rmse_final) !== null) meta.push(["RMSE", fmt(num(mf.rmse_final))]);
  if (num(mf.mape_final) !== null) meta.push(["MAPE", `${fmt(num(mf.mape_final), 1)}%`]);
  if (num(mf.mae_persistencia) !== null) meta.push(["MAE persistencia", fmt(num(mf.mae_persistencia))]);
  if (num(mf.mae_media_movil) !== null) meta.push(["MAE media móvil", fmt(num(mf.mae_media_movil))]);

  const pct = (v) => (num(v) === null ? "—" : `${fmt(num(v), 1)} %`);
  const ph = (v) => (num(v) === null ? "—" : fmt(num(v), 3));

  $("hg-card").innerHTML = `
    <article class="verdict">
      <p class="verdict-claim">${esc(hg.texto ?? "—")}</p>
      <div class="verdict-main">
        <div class="score">
          <span class="score-num">${fmt(r2)}</span>
          <span class="score-label">R² en prueba (hold-out)${icTxt}</span>
        </div>
        ${track}
      </div>
      <ul class="verdict-status">
        <li>${chipEstado(a.estado)} <b>HG-a (eficacia):</b> criterio de referencia R² ≥ ${crit.toFixed(2)} (decisión por IC95: se rechaza si el límite superior queda por debajo).</li>
        <li>${chipEstado(b.estado)} <b>HG-b (baselines):</b> el MAE baja ${pct(b.reduccion_persistencia_pct)} frente a la persistencia (p Holm ${ph(b.p_holm_persistencia)}) y ${pct(b.reduccion_media_movil_pct)} frente a la media móvil (p Holm ${ph(b.p_holm_media_movil)}).</li>
      </ul>
      ${meta.length ? `<dl class="verdict-meta">${meta.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>` : ""}
      ${hg.evidencia ? `<p class="verdict-evidence"><b>Evidencia:</b> ${esc(hg.evidencia)}</p>` : ""}
    </article>`;
}

function renderHipotesisEspecificas(r) {
  const lista = Array.isArray(r.hipotesis_especificas) ? r.hipotesis_especificas : [];
  $("he-cards").innerHTML = lista.map((h, i) => {
    const n = dimNumero(h.codigo, i);
    const shap = num(h.importancia_shap);
    const imp = shap !== null ? shap : num(h.importancia_ganancia);
    const etiqImp = shap !== null ? "Importancia SHAP" : "Importancia (ganancia)";
    const ancho = imp !== null ? Math.max(0, Math.min(1, imp)) * 100 : 0;

    const niveles = (Array.isArray(h.niveles) ? h.niveles : []).map((nv) => {
      const d = num(nv.dMAE), p = num(nv.p_holm);
      const val = d === null && p === null
        ? `<span style="color:var(--muted,#57655C)">${esc(nv.nota || "—")}</span>`
        : `ΔMAE ${fmtSigned(d)} · p Holm ${fmt(p, 4)}`;
      return `<li style="display:flex;justify-content:space-between;gap:8px;font-size:12.5px;padding:3px 0"><span>${esc(nv.nivel)}</span><span>${val}</span></li>`;
    }).join("");

    return `
      <article class="dim-card dim-x${n}">
        <header class="dim-head">
          <span class="dim-index">${esc(h.codigo)}</span>
          ${chipEstado(h.estado)}
        </header>
        <p class="hyp-text">${esc(h.texto)}</p>
        <ul style="list-style:none;margin:0 0 10px;padding:0;border-top:1px solid #E9ECE8">${niveles}</ul>
        <dl class="dim-stats">
          <div><dt>${etiqImp}</dt><dd>${fmtPct(imp)}</dd></div>
        </dl>
        <div class="share" aria-hidden="true">
          <div class="share-track"><div class="share-fill" style="width:${ancho}%"></div></div>
          <span class="share-label">Peso de la dimensión en el modelo</span>
        </div>
      </article>`;
  }).join("");

  if (num(r.n_splits_ablation) !== null) N_SPLITS_TSS = num(r.n_splits_ablation);
  $("he-note").textContent = lista.length
    ? `El estado resume tres contrastes (validación temporal, prueba con LightGBM directo y modelo final). En validación se aplicó Wilcoxon unilateral con Holm sobre ${N_SPLITS_TSS} particiones expansivas (brecha de 2 meses); como los folds se solapan, el p-valor indica consistencia y no una probabilidad exacta de error tipo I. La contribución es predictiva, no causal.`
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