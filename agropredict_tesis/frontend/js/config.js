// config.js — Configuración de la URL de la API.

(function () {
  const host = window.location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "";

  // ⚠️ Cambia esta URL por la de tu proyecto desplegado en Render o Railway.
  // Asegúrate de que el backend en la nube tenga el nuevo modelo (h=3) y los CSVs actualizados.
  const PROD_URL = "https://tesis-titulo-unsch.onrender.com";

  // Exponer globalmente para que app.js la use
  window.API_BASE_URL = isLocal ? "http://localhost:8000" : PROD_URL;

  console.log("[AgroPredict] API_BASE_URL =", window.API_BASE_URL);
})();