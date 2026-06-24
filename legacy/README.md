# Legacy

Esta carpeta archiva material historico que no forma parte del flujo oficial
del MVP final.

## Contenido

- `original_experiments/`: experimentos, scripts, checkpoints, figuras y
  reportes generados durante fases previas del proyecto.
- `development/validated_mvp_build_notes/`: notas tecnicas y notebook de
  desarrollo conservados para trazabilidad.

## Regla De Uso

El punto de entrada oficial es:

```text
notebooks/01_final_mvp.ipynb
```

El paquete oficial es:

```text
src/trustworthy_credit/
```

Los archivos de `legacy/` no deben importarse ni ejecutarse para reproducir el
MVP final, salvo que se este auditando el historial del proyecto.
