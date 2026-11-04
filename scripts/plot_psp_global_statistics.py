"""One global-mask statistics comparison figure per PSP polar."""
import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT,configured_path,sha256
from plot_style import STYLE,WIDTH_MM,export

METRICS=[('minimum','Minimum'),('maximum','Maximum'),('mean','Mean'),
         ('q01',r'1\% quantile'),('q99',r'99\% quantile'),('retained_pixel_percent','Retained pixels')]
POLICIES=[('automatic','Automatic','#0072B2','o'),
          ('automatic_mask2','Automatic + Mask 2','#D55E00','s'),
          ('automatic_mask2_crop10','Automatic + Mask 2 + crop 10','#009E73','^')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-tex',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    base=configured_path('OUTPUT_DIR')/'psp'
    source=base/'working/global_statistics/global_mask_statistics.csv'
    rows=list(csv.DictReader(source.open()))
    polars=sorted({r['polar'] for r in rows})
    masks=sorted({r['mask'] for r in rows},key=lambda s:tuple(float(v) for v in s[1:].split('_t')))
    assert len(polars)==8 and len(masks)==9 and len(rows)==216
    lookup={(r['polar'],r['mask'],r['policy']):r for r in rows};assert len(lookup)==216
    limits={}
    for metric,_ in METRICS:
        values=np.array([float(r[metric]) for r in rows]);assert np.isfinite(values).all()
        lo,hi=values.min(),values.max();pad=.08*max(hi-lo,1e-6)
        limits[metric]=(lo-pad,hi+pad)
    limits['retained_pixel_percent']=(0,100)
    destination=args.output_dir or base/'exploratory/figures/global_statistics'
    destination.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    style={**STYLE,'text.usetex':not args.no_tex}
    labels=[s[1:].replace('_t',' / ') for s in masks]
    outputs=[]
    with plt.rc_context(style),PdfPages(destination/'all_polars.pdf') as combined:
        for polar in polars:
            fig,axes=plt.subplots(3,2,figsize=(WIDTH_MM['double']/25.4,180/25.4),sharey=True,layout='constrained')
            for ax,(metric,title) in zip(axes.flat,METRICS):
                for offset,(policy,label,color,marker) in zip([-.22,0,.22],POLICIES):
                    values=[float(lookup[polar,mask,policy][metric]) for mask in masks]
                    ax.scatter(values,np.arange(9)+offset,s=13,marker=marker,color=color,label=label,zorder=3)
                ax.set_title(title.replace(r'\%','%') if args.no_tex else title)
                ax.set_xlim(limits[metric]);ax.set_yticks(np.arange(9),labels,fontsize=8)
                ax.set_ylim(8.6,-.6);ax.grid(axis='x',alpha=.22)
                ax.set_xlabel('Retained pixels [percent]' if metric=='retained_pixel_percent' else r'$C_p$')
                ax.tick_params(axis='x',labelsize=8)
            for ax in axes[:,0]:ax.set_ylabel('Patch / threshold',fontsize=9)
            angle=float(next(r['angle_deg'] for r in rows if r['polar']==polar))
            fig.suptitle(rf'Polar {polar}: $\alpha={angle:g}^\circ$'+'\nGlobal statistics over all 3,639 snapshots',fontsize=11)
            handles,legend_labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,legend_labels,loc='outside lower center',ncol=1,fontsize=9,frameon=False)
            stem=destination/f'polar_{polar}'
            export(fig,stem);combined.savefig(fig);plt.close(fig)
            outputs.append(stem.name)
    provenance={'source_csv_sha256':sha256(source),'code_sha256':sha256(Path(__file__)),
        'polars':polars,'figures':outputs,'metrics':[m for m,_ in METRICS],
        'common_limits_per_metric':limits,'rendering':'mathtext' if args.no_tex else 'LaTeX',
        'size_mm':[180,180],'definitions':'Pooled equal-weight pixel-time Cp samples; no smoothing or normalization; offsets separate three policies within each categorical mask row.'}
    (destination/'figures.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Exported {len(outputs)} polar figures (PNG/PDF) and combined PDF to {destination}')

if __name__=='__main__':main()
