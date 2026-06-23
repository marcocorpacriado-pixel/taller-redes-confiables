# Plan de migracion hacia el MVP unificado

## Estado de partida

El MVP validado vive actualmente en:

```text
src/dani_credit/
notebooks/01_mvp_dani_professional.ipynb
tests/test_dani_uncertainty_regressions.py
```

La version estable esta marcada con:

```text
v1.0-mvp-defendible
```

La migracion debe preservar ese MVP y construir una arquitectura comun encima.

## Fase 1: Esqueleto comun

Objetivo: crear la estructura profesional sin mover codigo.

Acciones:

- Crear `src/trustworthy_credit/`.
- Crear `docs/`.
- Crear `reports/presentation/`.
- Documentar arquitectura y plan de migracion.

No se hace:

- No se borran carpetas antiguas.
- No se mueven modulos.
- No se tocan notebooks ejecutados.
- No se recalculan resultados.
- No se modifica la Pareto principal.
- No se modifica M2 de incertidumbre.

## Fase 2: Dani como base canonica

Objetivo: promover el pipeline POO validado como columna vertebral comun.

Mapeo previsto:

```text
src/dani_credit/data_contract.py  -> src/trustworthy_credit/data_contract.py
src/dani_credit/preprocessing.py  -> src/trustworthy_credit/preprocessing.py
src/dani_credit/splitting.py      -> src/trustworthy_credit/splitting.py
src/dani_credit/layers.py         -> src/trustworthy_credit/layers.py
src/dani_credit/models.py         -> src/trustworthy_credit/models.py
src/dani_credit/metrics.py        -> src/trustworthy_credit/metrics.py
src/dani_credit/tuning.py         -> src/trustworthy_credit/tuning.py
src/dani_credit/uncertainty.py    -> src/trustworthy_credit/uncertainty.py
```

Estrategia segura:

1. Empezar con imports puente o copias controladas.
2. Actualizar tests para cubrir el paquete unificado.
3. Mantener `src/dani_credit/` hasta que el nuevo paquete pase todas las
   validaciones.

Lo que no se sustituye:

- Pareto principal validada.
- M2 corregido.
- Preprocesamiento con `EXT_NULL_COUNT` crudo.
- Tests de regresion de incertidumbre.

## Fase 3: Notebook maestro

Objetivo: crear `notebooks/01_main_mvp.ipynb`.

Base tecnica:

- Notebook profesional de Dani.
- Clases POO del paquete unificado.

Aportacion de Javi:

- Orden narrativo.
- EDA compacta.
- Explicacion pedagogica de arquitectura, FAIR loss e incertidumbre.
- Tabla final clara.

Regla:

El notebook maestro debe orquestar clases. No debe contener funciones largas ni
duplicar logica de `src/`.

## Fase 4: Integracion de Marco

Objetivo: incorporar robustez experimental y visualizaciones sin reemplazar el
MVP validado.

Elementos a rescatar:

- `src/visualization.py`: adaptar graficas a `reporting.py` y `plots.py`.
- `src/uncertainty.py::mc_dropout_predict`: convertir en
  `MCDropoutUncertaintyEstimator`.
- `pareto_experiment.py`: convertir la idea en `MultiSeedParetoRunner`.
- Experimentos M0-M6: usarlos como narrativa historica o comparativa, no como
  sustituto del pipeline final.
- Features extendidas: reservar para modulo futuro `features.py`.

Elementos a no promover directamente:

- Scripts procedurales como flujo principal.
- Checkpoints antiguos como fuente canonica.
- Resultados que usen protocolo distinto sin documentarlo.

## Fase 5: Integracion de Javi

Objetivo: mejorar claridad de entrega y anadir una incertidumbre complementaria.

Elementos a rescatar:

- Estructura narrativa del notebook.
- EDA inicial sobre target, genero y missingness de `EXT_SOURCE`.
- Presentacion compacta de resultados.
- MC Dropout como contraste frente a M2.

Elementos a no promover directamente:

- Dependencias no integradas como `src/fair_credit.py` si no estan en el repo.
- Funciones largas dentro del notebook.
- Cambio global a backend JAX como decision por defecto.
- Pareto o seleccion de hiperparametros calculada con test.

## Fase 6: Extras de nota alta

Orden recomendado:

1. MC Dropout como comparacion secundaria.
2. Pareto multi-semilla.
3. OOF uncertainty para M2 mas honesto.
4. Enriquecimiento relacional con tablas auxiliares de Home Credit.
5. Calibracion, Brier score, ECE o analisis top-k de incertidumbre.

Estos extras deben vivir en ramas separadas y no sustituir el MVP principal
hasta que pasen revision.

## Criterios para borrar o archivar codigo antiguo

Solo se limpiara codigo fuera de la estructura unificada cuando:

1. La funcionalidad este migrada.
2. Haya tests equivalentes.
3. El notebook maestro ejecute.
4. El README apunte al flujo nuevo.
5. El equipo valide que no se pierde ninguna aportacion relevante.

Hasta entonces, lo antiguo se conserva como referencia historica.
