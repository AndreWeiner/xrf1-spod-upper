"""Shared exploratory/paper figure dimensions and typography."""
STYLE = {
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'axes.titlesize': 10, 'axes.linewidth': 0.8,
    'lines.linewidth': 1.4, 'savefig.dpi': 300, 'text.usetex': True,
}
WIDTH_MM = {'single': 89, 'double': 180}


def export(fig, stem):
    for suffix in ('.pdf', '.png'):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
