# Legacy

Esta carpeta archiva material histórico del proyecto. Se conserva por
trazabilidad, pero no forma parte del flujo oficial de ejecución ni debe
interpretarse como una alternativa activa al entregable final.

## Qué Contiene

- `original_experiments/`: experimentos, scripts, checkpoints, figuras, pesos,
  notebooks y reportes generados durante fases previas del proyecto.
- `development/validated_mvp_build_notes/`: notas técnicas y un notebook de
  desarrollo conservados para documentar cómo evolucionó la solución.

Parte de este material puede contener imports, rutas o nombres de paquetes
anteriores, por ejemplo `src.uncertainty` o `src/dani_credit`. También puede
incluir notebooks ejecutados fuera del flujo actual, checkpoints antiguos,
figuras históricas y scripts que dependían de una estructura de carpetas distinta.
Eso es esperable dentro de `legacy/` y no describe el estado final del proyecto.

## Flujo Oficial

El MVP oficial es:

```text
notebooks/01_final_mvp.ipynb
```

Los extras oficiales son:

```text
notebooks/02_extras_grade10.ipynb
```

El paquete Python oficial es:

```text
src/trustworthy_credit/
```

## Regla De Uso

Los archivos de `legacy/` no deben importarse ni ejecutarse para reproducir el
MVP final. Su función es explicar el historial del trabajo, no generar los
resultados oficiales.

Si se consulta esta carpeta, debe hacerse como archivo histórico: para entender
decisiones previas, comparar enfoques antiguos o recuperar contexto de diseño.
La reproducción del entregable debe hacerse siempre desde los notebooks y el
paquete oficiales indicados arriba.
