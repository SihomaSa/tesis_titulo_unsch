// Cambia esto por la URL real de tu API cuando despliegues
// (por ejemplo Railway, Render, o Cloud Run). En local, deja localhost.
// Detectar entorno

const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);

const API_BASE_URL = isLocal
  ? "http://localhost:8000"
  : "https://tesistitulounsch-production.up.railway.app";

export default API_BASE_URL;