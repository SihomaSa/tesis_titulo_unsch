// Cambia esto por la URL real de tu API cuando despliegues
// (por ejemplo Railway, Render, o Cloud Run). En local, deja localhost.
// Detectar entorno

(function () {
  const host = window.location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "";

  // ⚠️ Cambia esta URL por la de tu proyecto en Railway
  const RAILWAY_URL = "https://tesistitulounsch-production.up.railway.app";

  // Exponer globalmente para que app.js la use
  window.API_BASE_URL = isLocal ? "http://localhost:8000" : RAILWAY_URL;

  console.log("[AgroPredict] API_BASE_URL =", window.API_BASE_URL);
})();