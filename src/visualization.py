"""
visualization.py
────────────────
Funciones de visualización para el proyecto Taller B4-T1.

Todas las funciones:
  · Muestran el gráfico por pantalla (plt.show())
  · Lo guardan en PLOTS_DIR si se pasa save=True
  · Devuelven la figura para poder encadenar o modificar

Uso típico:
    from src.visualization import plot_curves, plot_pareto, savefig
    plot_curves(history, 'M3', save=True)
    plot_pareto(df_pareto, save=True)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

sns.set_theme(style='whitegrid')

PLOTS_DIR = 'report_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

_INVALID_CHARS = str.maketrans({
    '*': '', '?': '', '"': '', '<': '', '>': '',
    '|': '', ':': '', '\\': '', '/': '',
    '—': '-', '–': '-', ' ': '_', '+': 'plus',
})


def _clean_filename(name: str) -> str:
    return name.translate(_INVALID_CHARS)


def savefig(filename: str, fig=None):
    """Guarda la figura activa (o la pasada) en PLOTS_DIR."""
    filename = _clean_filename(filename)
    path = os.path.join(PLOTS_DIR, filename)
    (fig or plt).savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 Guardado: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CURVAS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def plot_curves(history, model_name: str,
                zoom_n: int = 20, save: bool = False):
    """
    Curvas de pérdida y AUC con zoom en las últimas épocas.

    Parámetros:
      history    : objeto History de Keras (o _FakeHistory del checkpoint)
      model_name : nombre del modelo para el título
      zoom_n     : número de épocas a mostrar en el zoom
      save       : si True guarda el PDF en report_plots/
    """
    n  = len(history.history['loss'])
    zs = max(0, n - zoom_n)
    ep = list(range(1, n + 1))

    best_auc   = max(history.history['val_auc'])
    best_epoch = history.history['val_auc'].index(best_auc) + 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f'{model_name}\nMejor AUC val = {best_auc:.4f}  '
        f'(época {best_epoch}/{n})',
        fontsize=13, fontweight='bold'
    )

    for row, (metric, ylabel) in enumerate([('loss', 'BCE Loss'),
                                             ('auc',  'AUC-ROC')]):
        tr = history.history[metric]
        vl = history.history[f'val_{metric}']

        # Curva completa
        ax = axes[row, 0]
        ax.plot(ep, tr, color='steelblue', lw=1.5, label='Train')
        ax.plot(ep, vl, color='tomato',    lw=1.5, label='Val')
        ax.axvline(best_epoch, color='green', ls='--', lw=1.2,
                   label=f'Mejor val (é.{best_epoch})')
        ax.set(title=f'{ylabel} — Curva completa',
               xlabel='Época', ylabel=ylabel)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Zoom últimas épocas
        ax = axes[row, 1]
        ax.plot(ep[zs:], tr[zs:], color='steelblue', lw=1.5,
                marker='o', ms=3, label='Train')
        ax.plot(ep[zs:], vl[zs:], color='tomato', lw=1.5,
                marker='o', ms=3, label='Val')
        if best_epoch > zs:
            ax.axvline(best_epoch, color='green', ls='--', lw=1.2,
                       label=f'Mejor val (é.{best_epoch})')
        ax.set(title=f'{ylabel} — Zoom últimas {zoom_n} épocas',
               xlabel='Época', ylabel=ylabel)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        savefig(f'curves_{model_name.replace(" ", "_")}.pdf')
    plt.show()
    return fig, best_auc, best_epoch


def plot_curves_with_lr(history, model_name: str,
                        zoom_n: int = 30, save: bool = False,
                        filename: str = None):
    """
    Curvas de pérdida, AUC y evolución del learning rate.
    Útil para modelos con ReduceLROnPlateau.
    """
    n  = len(history.history['loss'])
    zs = max(0, n - zoom_n)
    ep = list(range(1, n + 1))

    best_auc   = max(history.history['val_auc'])
    best_epoch = history.history['val_auc'].index(best_auc) + 1
    lrs        = history.history.get('learning_rate', [None] * n)

    fig = plt.figure(figsize=(16, 11))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)
    fig.suptitle(
        f'{model_name}\nMejor AUC val = {best_auc:.4f}  '
        f'(época {best_epoch}/{n})',
        fontsize=13, fontweight='bold'
    )

    for row, (metric, ylabel) in enumerate([('loss', 'BCE Loss'),
                                             ('auc',  'AUC-ROC')]):
        tr = history.history[metric]
        vl = history.history[f'val_{metric}']

        ax = fig.add_subplot(gs[row, 0])
        ax.plot(ep, tr, color='steelblue', lw=1.5, label='Train')
        ax.plot(ep, vl, color='tomato',    lw=1.5, label='Val')
        ax.axvline(best_epoch, color='green', ls='--', lw=1.2,
                   label=f'Mejor val (é.{best_epoch})')
        ax.set(title=f'{ylabel} — Curva completa',
               xlabel='Época', ylabel=ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[row, 1])
        ax.plot(ep[zs:], tr[zs:], color='steelblue', lw=1.5,
                marker='o', ms=3, label='Train')
        ax.plot(ep[zs:], vl[zs:], color='tomato', lw=1.5,
                marker='o', ms=3, label='Val')
        if best_epoch > zs:
            ax.axvline(best_epoch, color='green', ls='--', lw=1.2,
                       label=f'Mejor val (é.{best_epoch})')
        ax.set(title=f'{ylabel} — Zoom últimas {zoom_n} épocas',
               xlabel='Época', ylabel=ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Learning rate
    ax_lr = fig.add_subplot(gs[2, :])
    if lrs[0] is not None:
        ax_lr.semilogy(ep, lrs, color='purple', lw=1.5)
        ax_lr.axvline(best_epoch, color='green', ls='--', lw=1.2)
        ax_lr.set(title='Evolución del Learning Rate (escala log)',
                  xlabel='Época', ylabel='LR')
        ax_lr.grid(alpha=0.3, which='both')
    else:
        ax_lr.text(0.5, 0.5, 'LR no registrado',
                   ha='center', va='center', transform=ax_lr.transAxes)

    if save:
        fname = filename if filename else f'curves_{model_name}_lr.pdf'
        savefig(fname)
    plt.show()
    return fig, best_auc, best_epoch


# ══════════════════════════════════════════════════════════════════════════════
# TABLA COMPARATIVA DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def print_model_table(resultados: list):
    """
    Imprime tabla comparativa de modelos.

    resultados: lista de dicts con keys:
      nombre, val_auc, test_auc, n_params, n_epochs
    """
    print(f"\n{'Modelo':<28} {'AUC val':>9} {'AUC test':>9} "
          f"{'Paráms':>9} {'Épocas':>7}")
    print('─' * 67)
    best_test = max(r['test_auc'] for r in resultados)
    for r in resultados:
        marca = ' ◄' if r['test_auc'] == best_test else ''
        print(f"{r['nombre']:<28} {r['val_auc']:>9.4f} {r['test_auc']:>9.4f} "
              f"{r['n_params']:>9,} {r['n_epochs']:>7}{marca}")


# ══════════════════════════════════════════════════════════════════════════════
# CURVA DE PARETO — FAIR LOSS
# ══════════════════════════════════════════════════════════════════════════════

def plot_pareto(df_pareto: pd.DataFrame, save: bool = False):
    """
    Curva de Pareto Fairness vs Precisión con barras de error.

    df_pareto debe tener columnas:
      lambda, auc, dp, mean_F, mean_M
      (opcional: auc_std, dp_std para barras de error con múltiples semillas)
    """
    LAMBDAS = df_pareto['lambda'].tolist()
    has_std = 'auc_std' in df_pareto.columns

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('FAIR Loss — Curva de Pareto  (base: M6)',
                 fontsize=14, fontweight='bold')

    # ── Scatter Pareto ────────────────────────────────────────────────────
    ax = axes[0]

    if has_std:
        ax.errorbar(
            df_pareto['dp'], df_pareto['auc'],
            xerr=df_pareto['dp_std'], yerr=df_pareto['auc_std'],
            fmt='o', color='steelblue', ecolor='lightblue',
            elinewidth=2, capsize=4, ms=10, zorder=5
        )
    else:
        ax.scatter(df_pareto['dp'], df_pareto['auc'],
                   s=150, color='steelblue', zorder=5)

    ax.plot(df_pareto['dp'], df_pareto['auc'],
            color='gray', alpha=0.4, ls='--')

    for _, row in df_pareto.iterrows():
        ax.annotate(f"λ={row['lambda']:.1f}",
                    xy=(row['dp'], row['auc']),
                    xytext=(6, 5), textcoords='offset points', fontsize=10)

    dp0 = df_pareto.loc[df_pareto['lambda'] == 0, 'dp'].values[0]
    ax.axvline(dp0, color='red', ls=':', lw=1.2, alpha=0.7,
               label=f'DP base λ=0 ({dp0:.3f})')

    ax.set_xlabel('DP Gap  |E[ŷ|G=M] − E[ŷ|G=F]|', fontsize=11)
    ax.set_ylabel('AUC-ROC (test)', fontsize=11)
    ax.set_title('Tradeoff Fairness–Precisión\n'
                 '(izq-arriba = óptimo en ambas dimensiones)', fontsize=11)
    if has_std:
        ax.set_title('Tradeoff Fairness–Precisión\n'
                     '(barras de error = std entre semillas)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Predicción media por género ───────────────────────────────────────
    ax2 = axes[1]
    x   = range(len(LAMBDAS))
    ax2.plot(x, df_pareto['mean_M'], 'o-', color='steelblue',
             lw=2, ms=8, label='Hombres (M)')
    ax2.plot(x, df_pareto['mean_F'], 's-', color='tomato',
             lw=2, ms=8, label='Mujeres (F)')
    ax2.fill_between(x, df_pareto['mean_M'], df_pareto['mean_F'],
                     alpha=0.15, color='gray', label='DP gap')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'λ={l}' for l in LAMBDAS], rotation=30)
    ax2.set_ylabel('P(impago) media predicha', fontsize=11)
    ax2.set_title('Convergencia de predicciones\nentre grupos al aumentar λ',
                  fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        savefig('08_pareto_fair.pdf')
    plt.show()
    return fig


def print_pareto_table(df_pareto: pd.DataFrame):
    """Imprime tabla resumen del experimento Pareto."""
    has_std    = 'auc_std' in df_pareto.columns
    auc_base   = df_pareto.loc[df_pareto['lambda'] == 0, 'auc'].values[0]
    dp_base    = df_pareto.loc[df_pareto['lambda'] == 0, 'dp'].values[0]

    if has_std:
        print(f"\n{'λ':>5} {'AUC':>8} {'±std':>7} {'DP gap':>8} "
              f"{'±std':>7} {'P̂(M)':>8} {'P̂(F)':>8} "
              f"{'ΔAUC':>8} {'Reduc.DP':>10}")
        print('─' * 72)
        for _, r in df_pareto.iterrows():
            print(f"{r['lambda']:>5.1f} {r['auc']:>8.4f} "
                  f"{r['auc_std']:>7.4f} {r['dp']:>8.4f} "
                  f"{r['dp_std']:>7.4f} {r['mean_M']:>8.4f} "
                  f"{r['mean_F']:>8.4f} "
                  f"{r['auc']-auc_base:>+8.4f} "
                  f"{(1-r['dp']/dp_base)*100:>9.1f}%")
    else:
        print(f"\n{'λ':>5} {'AUC':>8} {'DP gap':>8} "
              f"{'P̂(M)':>8} {'P̂(F)':>8} {'ΔAUC':>8} {'Reduc.DP':>10}")
        print('─' * 60)
        for _, r in df_pareto.iterrows():
            print(f"{r['lambda']:>5.1f} {r['auc']:>8.4f} {r['dp']:>8.4f} "
                  f"{r['mean_M']:>8.4f} {r['mean_F']:>8.4f} "
                  f"{r['auc']-auc_base:>+8.4f} "
                  f"{(1-r['dp']/dp_base)*100:>9.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# ARQUITECTURA DEL MODELO (sin graphviz)
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_arch(save: bool = False):
    """
    Diagrama manual de la arquitectura M6 (no requiere graphviz).
    """
    COLORS = {
        'input':   '#AED6F1', 'lambda':  '#A9DFBF',
        'custom':  '#F9E79F', 'dense':   '#D2B4DE',
        'dropout': '#FAD7A0', 'concat':  '#F1948A',
        'output':  '#85C1E9',
    }

    def draw_box(ax, x, y, w, h, label, sublabel, color, fs=10):
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.05',
            facecolor=color, edgecolor='gray', lw=1.2, zorder=3))
        ax.text(x, y + 0.07, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', zorder=4)
        ax.text(x, y - 0.22, sublabel, ha='center', va='center',
                fontsize=8, color='#555555', zorder=4, style='italic')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.5),
                    zorder=2)

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis('off')
    W, H = 3.2, 0.65

    draw_box(ax, 5, 15, 4, H, 'Input', 'shape: (None, 12)', COLORS['input'])
    for x in [2.5, 5.0, 7.5]:
        arrow(ax, x, 14.67, x, 13.85)

    draw_box(ax, 2.5, 13.5, W, H, 'Lambda',
             'extract_debt_ratio  (None, 1)', COLORS['lambda'])
    arrow(ax, 2.5, 13.17, 2.5, 12.35)
    draw_box(ax, 2.5, 12.0, W, H, 'DebtRatioLayer',
             r'σ(α·(x−θ))  |  2 params  (None, 1)', COLORS['custom'])

    draw_box(ax, 5.0, 13.5, W, H, 'Dense (128)',
             'ReLU  (None, 128)', COLORS['dense'])
    arrow(ax, 5.0, 13.17, 5.0, 12.35)
    draw_box(ax, 5.0, 12.0, W, H, 'Dropout (0.30)',
             'p=0.30  (None, 128)', COLORS['dropout'])
    arrow(ax, 5.0, 11.67, 5.0, 10.85)
    draw_box(ax, 5.0, 10.5, W, H, 'Dense (64)',
             'ReLU  (None, 64)', COLORS['dense'])
    arrow(ax, 5.0, 10.17, 5.0, 9.35)
    draw_box(ax, 5.0, 9.0,  W, H, 'Dropout (0.20)',
             'p=0.20  (None, 64)', COLORS['dropout'])

    draw_box(ax, 7.5, 13.5, W, H, 'Lambda',
             'extract_ext_sources  (None, 3)', COLORS['lambda'])
    arrow(ax, 7.5, 13.17, 7.5, 12.35)
    draw_box(ax, 7.5, 12.0, W, H, 'ExtSourceLayer',
             r'σ(w₁s₁+w₂s₂+w₃s₃+b)  |  4 params  (None, 1)',
             COLORS['custom'])

    arrow(ax, 2.5, 11.67, 2.5, 7.60)
    arrow(ax, 5.0,  8.67, 5.0, 7.60)
    arrow(ax, 7.5, 11.67, 7.5, 7.60)

    draw_box(ax, 5.0, 7.25, 5.5, H, 'Concatenate',
             '[Dense64  ||  DebtLayer  ||  ExtLayer]  (None, 66)',
             COLORS['concat'])
    arrow(ax, 5.0, 6.92, 5.0, 6.10)
    draw_box(ax, 5.0, 5.75, W, H, 'Dense (1)',
             'sigmoid  (None, 1)', COLORS['output'])

    ax.set_title('Arquitectura M6 — Dual Custom + Dropout\n'
                 'Total parámetros: 9,993  |  Entrenables: 9,993',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(handles=[
        mpatches.Patch(facecolor=COLORS[k], label=v)
        for k, v in [('input','Entrada'), ('lambda','Lambda'),
                     ('custom','Capa Custom'), ('dense','Dense'),
                     ('dropout','Dropout'), ('concat','Concatenate'),
                     ('output','Salida')]
    ], loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save:
        savefig('arch_M6_DualCustom.pdf')
        savefig('arch_M6_DualCustom.png')
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INCERTIDUMBRE
# ══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_by_class(var_m6, var_fair, y_te,
                               save: bool = False):
    """KDE de varianza MC Dropout por clase TARGET."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Distribución de Incertidumbre (Varianza MC Dropout)\n'
                 'por clase TARGET', fontsize=14, fontweight='bold')

    for ax, (var, mname) in zip(axes, [(var_m6,  'M6 — Base (sin FAIR)'),
                                        (var_fair, 'M6 — FAIR (λ=0.5)')]):
        sns.kdeplot(var[y_te == 0], ax=ax, fill=True, alpha=0.5,
                    color='steelblue', label='TARGET=0 (pagó)')
        sns.kdeplot(var[y_te == 1], ax=ax, fill=True, alpha=0.5,
                    color='tomato',    label='TARGET=1 (impago)')
        for t, c in [(0, 'steelblue'), (1, 'tomato')]:
            m = np.median(var[y_te == t])
            ax.axvline(m, color=c, ls='--', lw=1.5,
                       label=f'Mediana T={t}: {m:.5f}')
        ax.set_xlim(0, np.percentile(var, 99))
        ax.set(xlabel='Varianza (incertidumbre)', ylabel='Densidad',
               title=mname)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        savefig('09_uncertainty_by_class.pdf')
    plt.show()
    return fig


def plot_uncertainty_vs_missing(var_m6, var_fair, n_missing,
                                 save: bool = False):
    """Varianza media por número de EXT_SOURCE ausentes."""
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Incertidumbre vs Calidad de EXT_SOURCE\n'
                 '(0 = info completa, 3 = todo imputado)',
                 fontsize=13, fontweight='bold')

    for ax, (var, mname) in zip(axes, [(var_m6,  'M6 Base'),
                                        (var_fair, 'FAIR λ=0.5')]):
        medias = [var[n_missing == k].mean() for k in range(4)]
        stds   = [var[n_missing == k].std()  for k in range(4)]
        counts = [(n_missing == k).sum()      for k in range(4)]
        errors = [1.96 * s / np.sqrt(c) if c > 0 else 0
                  for s, c in zip(stds, counts)]

        ax.bar(range(4), medias, yerr=errors, capsize=5,
               color=colors, alpha=0.85,
               error_kw=dict(elinewidth=1.5))
        for i, (m, c) in enumerate(zip(medias, counts)):
            ax.text(i, m + errors[i] * 1.1, f'n={c:,}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['0 aus.', '1 aus.', '2 aus.', '3 aus.'])
        ax.set(ylabel='Varianza media (± IC 95%)', title=mname)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        savefig('10_uncertainty_vs_missing.pdf')
    plt.show()
    return fig