import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from matplotlib_venn import venn2, venn2_circles
from matplotlib.patches import PathPatch
from matplotlib import ticker
import seaborn as sns
from collections import OrderedDict
from abnumber import Chain

def plot_roc_curve(true_values, predictions, ax=None, title=None, label=None, lw=1, add_auc=True, baseline=True, figsize=(5, 5), verbose=True, **kwargs):
    """
    Plot ROC curve of a single model. Can be called repeatedly with same axis to plot multiple curves.
    :param true_values: Series of true values
    :param predictions: Series of prediction values
    :param ax: Use given axis (will create new one if None)
    :param title: Plot title
    :param label: ROC curve label
    :param lw: Line width
    :param add_auc: Add AUC value to label
    :param baseline: Plot baseline that indicates performance of random model (AUC 0.5)
    :param figsize: Figure size
    :param verbose: Print ROC to stdout
    :param kwargs: Additional arguments for plotting function
    :return: Figure axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    fpr, tpr, _ = metrics.roc_curve(true_values, predictions)
    roc_auc = safe_roc_auc_score(true_values, predictions)
    if pd.isna(roc_auc):
        print(f'Not enough classes to plot ROC of split: {label}')
        return ax
    label_auc = (label+': ' if label else '') + f'{roc_auc:.3f} AUC'
    if verbose:
        print(label_auc)
    ax.plot(fpr, tpr, lw=lw, label=label_auc if add_auc else label, **kwargs)
    if baseline:
        ax.plot([0, 1], [0, 1], color='grey', lw=1, label='Random baseline', linestyle='--')
    if title:
        ax.set_title(title)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', frameon=False)
    return ax

def safe_roc_auc_score(y_true, y_score, **kwargs):
    if set(np.unique(y_true).astype(np.int)) != set([0, 1]):
        return np.nan
    return metrics.roc_auc_score(y_true, y_score, **kwargs)

def plot_precision_recall_curve(true_values, predictions, sample_weight=None, ax=None, title=None, label=None, baseline=True, **kwargs):
    """
    Plot Precision-Recall curve of a single model. Can be called repeatedly with same axis to plot multiple curves.
    :param true_values: Series of true values
    :param predictions: Series of prediction values
    :param sample_weight: Weights for each sample
    :param ax: Use given axis (will create new one if None)
    :param title: Plot title
    :param label: ROC curve label
    :param baseline: Print random prediction baseline 
    :param kwargs: Additional arguments for plotting function
    :return: Figure axis
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    precision, recall, _ = metrics.precision_recall_curve(true_values, predictions, sample_weight=sample_weight)
    avg_precision = metrics.average_precision_score(true_values, predictions, sample_weight=sample_weight)
    avgpr_label = (label + ': ' if label else '') + f'{avg_precision:.3f} AvgPrec'
        
    ax.step(recall, precision, where='post', label=avgpr_label, **kwargs)
    
    if baseline:
        ax.axhline(np.mean(true_values), ls='--', label='Random baseline', color='lightgrey')
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc='lower left', frameon=False)

    return ax

def read_oasis_chains(oasis_path, index=None):
    sheets = pd.read_excel(oasis_path, sheet_name=None)
    vh_table = sheets['VH'].set_index('Antibody')
    vl_table = sheets['VL'].set_index('Antibody')
    scheme = vh_table['scheme'].iloc[0]
    vh_chains = Chain.from_dataframe(vh_table, scheme=scheme, as_series=True)    
    vl_chains = Chain.from_dataframe(vl_table, scheme=scheme, as_series=True)
    
    vh_germlines = []
    vl_germlines = []
    for vh_chain, vl_chain in zip(vh_chains, vl_chains):
        positions = sheets[vh_chain.name]
        vh_positions = positions[positions['Chain type'] == 'H']
        vl_positions = positions[positions['Chain type'].isin(['K', 'L'])]
        vh_germlines.append(vh_chain.clone(''.join(vh_positions['Germline'])))
        vl_germlines.append(vl_chain.clone(''.join(vl_positions['Germline'])))
    vh_germlines = pd.Series(vh_germlines, vh_chains.index)
    vl_germlines = pd.Series(vl_germlines, vl_chains.index)
    
    if index is not None:
        return vh_chains.reindex(index), vl_chains.reindex(index), vh_germlines.reindex(index), vl_germlines.reindex(index)
    return vh_chains, vl_chains, vh_germlines, vl_germlines

def plot_overlap(overlap, colors, figsize=(2.5, 3), baseline_label_color='white', predicted_label_color='white'):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    plot_overlap_venn(overlap.mean(), ax=axes[0], colors=colors, 
                      baseline_label_color=baseline_label_color, predicted_label_color=predicted_label_color)
    plot_overlap_boxplot(overlap, ax=axes[1], colors=colors)
    fig.tight_layout(pad=0);


def plot_overlap_boxplot(overlap, ax, colors):
    overlap = overlap.copy()
    overlap.columns = overlap.columns.str.replace(' ','\n')
    colors = [colors['Baseline'], '#ffdd00', colors['Prediction']]
    ax = sns.boxplot(
        ax=ax, 
        orient='v', 
        data=overlap.melt(value_name='Mutations'), 
        y='Mutations', 
        x='variable', 
        palette=colors, 
        order=overlap.columns
    )
    ax.set_xlabel('')
    ax.set_ylim(0, None)


def plot_overlap_venn(overlap, ax, colors, title=None, vertical_title=False, normalize_to=1, fontsize=12, fontweight=None,
                     baseline_label_color='white', predicted_label_color='white', outline='black'):
    sets = {
        '10': overlap['Baseline only'], 
        '01': overlap['Predicted only'], 
        '11': overlap['Shared']
    }
    is_float = overlap.sum() != float(int(overlap.sum()))

    v = venn2(
        sets, 
        alpha=0.9, 
        set_labels=None, 
        set_colors=(colors['Baseline'], colors['Prediction']), 
        subset_label_formatter='{:.1f}'.format if is_float else str,
        normalize_to=normalize_to,
        ax=ax
    )
    if outline:
        venn2_circles(sets, linewidth=1, color=outline, ax=ax, normalize_to=normalize_to)
    # blank plot to apply normalize_to in above plots
    venn2_circles(sets, linewidth=0, color='black', ax=ax, normalize_to=0.9)

    if overlap['Shared']:
        v.get_patch_by_id('11').set_color('#ddcc00')
        v.get_label_by_id('11').set_fontsize(fontsize)
        if fontweight:
            v.get_label_by_id('11').set_fontweight(fontweight)
    v.get_label_by_id('10').set_color(baseline_label_color)
    v.get_label_by_id('01').set_color(predicted_label_color)
    v.get_label_by_id('10').set_fontsize(fontsize)
    v.get_label_by_id('01').set_fontsize(fontsize)
    if fontweight:
        v.get_label_by_id('10').set_fontweight(fontweight)
        v.get_label_by_id('01').set_fontweight(fontweight)
        
    if title:
        if vertical_title:
            ax.axis('on')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_ylabel(title, visible=True, fontsize=fontsize)
        else:
            ax.set_title(title, fontsize=fontsize, pad=0)
    
def plot_overlap_venns(overlap, rows, cols, colors, figsize=(5, 3), fontsize=10, plot_mean=False, vertical_title=False,
                     baseline_label_color='white', predicted_label_color='white'):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for ax in axes.reshape(-1):
        ax.set_visible(False)
        
    max_norm = overlap.sum(axis=1).max()
        
    for (title, o), ax in zip(overlap.iterrows(), axes.reshape(-1)):
        ax.set_visible(True)
        normalize_to = o.sum() / max_norm * 1.1
        plot_overlap_venn(o, normalize_to=normalize_to, ax=ax, colors=colors, title=title, 
                          fontsize=fontsize, vertical_title=vertical_title, 
                          baseline_label_color=baseline_label_color,
                          predicted_label_color=predicted_label_color)
    
    if plot_mean:
        ax = axes.flatten()[-1]
        ax.set_visible(True)
        o = overlap.mean()
        normalize_to = o.sum() / max_norm * 1.1
        plot_overlap_venn(o, normalize_to=normalize_to, ax=ax, colors=colors, title='Mean', 
                          fontsize=fontsize, fontweight='bold', vertical_title=vertical_title, 
                          baseline_label_color=baseline_label_color,
                          predicted_label_color=predicted_label_color)

    fig.tight_layout(w_pad=0, h_pad=0 if vertical_title else 1)

    
OASIS_THRESHOLDS = {
    'loose': '1%',
    'relaxed': '10%',
    'medium': '50%',
    'strict': '90%'
}
    
    

def plot_oasis_curves(curves, groupby=None, ax=None, errorbars=False, thresholds=True, ncol=4, mark=True,
                      figsize=(6.5, 4.5), colors=None, lw=1.5, alpha=1, legend=True, markers='osx+vD<>^', fill_alpha=0.1, ylim=(0, 1), linestyles={}, linewidths={}):
    with sns.axes_style('whitegrid'):
        columns = [c for c in curves.columns if '%' in c and c != '0%']
        if groupby is not None:
            grouped = curves.groupby(groupby)
            curves_mean = grouped.mean()[columns]
            curves_low = grouped.quantile(0.25)[columns]
            curves_high = grouped.quantile(0.75)[columns]
        else:
            curves_mean = curves[columns]
            curves_low, curves_high = None, None
        if colors is None:
            colors = {group: None for group in curves_mean.index}
        x = pd.Series(range(len(curves_mean.columns)))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        xticks = [0] + list(range(9, len(curves_mean.columns)+1, 10))
        for i, (group, color) in enumerate(colors.items()):
            args = {}
            if color is not None:
                args['color'] = color
            y = curves_mean.loc[group]
            ls = linestyles.get(group, '-')
            group_lw = linewidths.get(group, lw)
            if mark:
                marker = markers[i]
                ax.scatter(xticks, y.iloc[xticks], color=color, marker=marker, s=30, zorder=2000, alpha=alpha);
            else:
                marker = None
            ax.plot(x, y, color=color, ls=ls, label=group, marker=marker, markevery=1000, lw=group_lw, zorder=1000, alpha=alpha);
            if curves_low is not None:
                ylow = curves_low.loc[group]
                yhigh = curves_high.loc[group]
                if errorbars:
                    ax.errorbar(
                        x.iloc[xticks], 
                        y.iloc[xticks], 
                        yerr=[
                            y.iloc[xticks] - ylow.iloc[xticks], 
                            yhigh.iloc[xticks] - y.iloc[xticks]
                        ], 
                        alpha=0.7, color=color, ls=ls, lw=0, elinewidth=group_lw, capsize=4);
                ax.fill_between(x, ylow, yhigh + 0.0000001, alpha=fill_alpha, **args)

        ax.set_xlabel('Human subject prevalence threshold')
        ax.set_xticks(xticks)
        ax.set_xticklabels(curves_mean.columns[i]+' ' for i in ax.get_xticks())
        ax.set_ylabel('OASis identity'); 
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        ax.set_ylim(ylim)
        ax.set_xlim(xticks[0]-2, xticks[-1]+2)
        if legend:
            ax.legend(loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 1), frameon=False, columnspacing=1, handletextpad=0.5, handlelength=1.5)

        if thresholds:
            for label, threshold in OASIS_THRESHOLDS.items():
                x = curves_mean.columns.get_loc(threshold)
                ax.axvline(x, color='#777777', lw=1)
                if label == 'strict':
                    ax.text(x-0.6, ylim[0]+0.03, label, rotation=90, color='#111111', ha='right')
                else:
                    ax.text(x+0.6, ylim[0]+0.03, label, rotation=90, color='#111111')
            
            
def plot_oasis_curve_and_barplots(oasis_curves, groupby, colors):
    fig, axes = plt.subplots(1, 1+len(OASIS_THRESHOLDS), figsize=(13.5, 4.5), gridspec_kw=dict(width_ratios=[3.5] + [1] * len(OASIS_THRESHOLDS)))

    ax = axes[0]
    plot_oasis_curves(oasis_curves, groupby=groupby, errorbars=False, ax=ax, colors=colors)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.15), frameon=False, columnspacing=1, handletextpad=0.5)

    for i, (label, threshold) in enumerate(OASIS_THRESHOLDS.items()):
        ax = axes[i+1]
        sns.boxplot(
            data=oasis_curves, 
            x=groupby, y=threshold, 
            order=colors.keys(), palette=colors, 
            width=0.95, 
            linewidth=1.5,
            ax=ax
        );
        adjust_box_widths(ax, 0.80)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        ax.set_ylabel(f'OASis identity ({label})')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('')

    fig.tight_layout();
    

def adjust_box_widths(ax, fac):
    """Adjust widths of seaborn boxplots boxes"""
    
    # iterating through axes artists:
    for c in ax.get_children():

        # searching for PathPatches
        if isinstance(c, PathPatch):
            # getting current width of box:
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)

            # setting new width of box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            # setting new width of median line
            for l in ax.lines:
                if np.array(l.get_xdata()).tolist() == [xmin, xmax]:
                    l.set_xdata([xmin_new, xmax_new])


def annotate_conservation(parental, pred, germline):
    if parental == pred:
        if parental == germline:
            return 'Conserved'
        return 'Back-mutated'
    if pred == germline:
        return 'Humanized'
    return 'Engineered'


def collect_position(pos, parental, pred, germline, name):
    return {
        'name': name,
        'pos': str(pos),
        'chain': 'Heavy' if pos.is_heavy_chain() else 'Light',
        'region': pos.get_region(),
        'in_vernier': pos.is_in_vernier(),
        'parental': parental,
        'predicted': pred,
        'germline': germline,
        'conservation': annotate_conservation(parental, pred, germline)
    }
    
    
def collect_positions(parental_chains, pred_chains, germline_chains):
    positions = []
    skipped = []
    for parental, pred, germline in zip(parental_chains, pred_chains, germline_chains):
        if pd.isna(pred):
            skipped.append(parental.name)
            continue
        assert parental.name == pred.name
        positions += [collect_position(pos, a, b, c, name=parental.name) for pos, (a, b, c) in parental.align(pred, germline)]
    if skipped:
        print('Skipped', skipped)
    return pd.DataFrame(positions)