#!/usr/bin/env bash
# Ejecutar desde agropredict_tesis/backend  (Git Bash). Borra SOLO restos obsoletos y archivos temporales.
set -e
cd "$(dirname "$0")" 2>/dev/null || true
echo "== Restos a borrar =="
for f in \
  "outputs/figures/figura_15_importancia_por_dimension.png" \
  outputs/tesis_texto/~\$*.docx
do
  for g in $f; do [ -e "$g" ] && { echo " - $g"; rm -f "$g"; }; done
done
# Tablas obsoletas (por si reaparecen). NO se toca tabla_comparacion_validacion.csv (la genera el pipeline vigente).
for t in tabla_4.1_comparacion_modelos tabla_4.4_wilcoxon_hipotesis tabla_11_rendimiento_por_frecuencia_cultivo tabla_9_rendimiento_provincial; do
  [ -e "outputs/tables/$t.csv" ] && { echo " - outputs/tables/$t.csv"; rm -f "outputs/tables/$t.csv"; }
done
echo "Listo. Cierra Word antes de correr main.py (el archivo ~\$*.docx indica que estaba abierto)."
