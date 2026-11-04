"""Separate fixed-taper and adaptive-tolerance comparisons from cached spectra."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT,configured_path
from plot_style import STYLE,WIDTH_MM,export
from reference_scales import strouhal, numerical_zoom_max


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-tex',action='store_true')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env',override=False)
    config=json.loads((ROOT/'config/spod_study.json').read_text())
    output=configured_path('OUTPUT_DIR')/'analysis'
    destination=output/'exploratory/figures/spod';destination.mkdir(parents=True,exist_ok=True)
    cases=['cp','cf_xy','cf_xyz']
    titles=[r'$c_p$',r'$(c_{fx},c_{fy})$',r'$(c_{fx},c_{fy},c_{fz})$']
    groups=[[(f'fixed_{k}',rf'$K={k}$') for k in config['fixed_tapers']],
        [('adaptive_1e-05',r'$\epsilon=10^{-5}$'),('adaptive_5e-05',r'$\epsilon=5\times10^{-5}$'),('adaptive_0.0001',r'$\epsilon=10^{-4}$')]]
    loaded={}
    for case in cases:
        for group in groups:
            for label,_ in group:
                p=output/'working/spod'/case/label
                record=json.loads((p/'run.json').read_text())
                assert record['status']=='complete' and record['rank']==400
                with np.load(p/'spectra.npz') as data:
                    loaded[(case,label)]={name:data[name].copy() for name in ['frequency','eigvals']}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    styles=['-','--','-.',':']
    for quantity,zoom,stem in [('leading',False,'taper_comparison_leading'),
                             ('leading',True,'taper_comparison_leading_zoom'),
                             ('sum',False,'taper_comparison_total')]:
        with plt.rc_context({**STYLE,'text.usetex':not args.no_tex}):
            fig,axes=plt.subplots(3,2,figsize=(WIDTH_MM['double']/25.4,180/25.4),layout='constrained',sharex=True)
            for row,(case,title) in enumerate(zip(cases,titles)):
                extrema=[]
                for column,group in enumerate(groups):
                    ax=axes[row,column]
                    for i,(label,legend) in enumerate(group):
                        data=loaded[(case,label)];frequency=strouhal(data['frequency'],'numerical')
                        values=data['eigvals'][:,0] if quantity=='leading' else data['eigvals'].sum(1)
                        valid=(frequency>0)&(values>0)
                        if zoom:valid &= frequency<=numerical_zoom_max()
                        extrema.extend(values[valid])
                        ax.plot(frequency[valid],values[valid],label=legend,color=colors[i],ls=styles[i],lw=1.25)
                    ax.set_yscale('log')
                    if not zoom:ax.set_xscale('log')
                    else:ax.set_xlim(0,numerical_zoom_max())
                    ax.set_title(title+(': fixed tapers' if column==0 else ': adaptive tapers'))
                    ax.grid(alpha=.2)
                    if row==0:ax.legend(loc='lower left',fontsize=8,ncol=2 if column==0 else 1)
                    if column==0:ax.set_ylabel(r'$\lambda_1$' if quantity=='leading' else r'$\sum_j\lambda_j$')
                    if row==2:ax.set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$')
                limits=(min(extrema)/1.35,max(extrema)*1.35)
                for ax in axes[row]:ax.set_ylim(*limits)
            export(fig,destination/stem);plt.close(fig)
    print('Exported leading, peak-band and summed-spectrum comparisons without recomputing SPOD.')


if __name__=='__main__':main()
