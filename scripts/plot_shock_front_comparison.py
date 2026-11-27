"""Compare saved smoothed shock fronts and pressure-based separation markers."""
import argparse
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path, sha256
from plot_style import STYLE, export


def normalized_front(front, coordinates):
    """Normalize against the same local section boundaries used in extraction."""
    coordinates=coordinates/1000
    # Saved front rows correspond to the original/refined coordinate rows.
    x=coordinates[...,0]
    span=-front[:,1]
    grid_span=-coordinates[...,1]
    eta=(span-grid_span.min())/(grid_span.max()-grid_span.min())
    chord_fraction=(front[:,0]-x[:,0])/(x[:,-1]-x[:,0])
    return eta,chord_fraction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-tex',action='store_true')
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    base=configured_path('OUTPUT_DIR')/'psp'
    shock_dir=base/'paper/mean_shocks';separation_dir=base/'paper/trailing_edge_separation'
    numerical_dir=base/'paper/numerical_overview'
    shock_record=json.loads((shock_dir/'extraction.json').read_text())
    separation_record=json.loads((separation_dir/'extraction.json').read_text())
    numerical_record=json.loads((numerical_dir/'preparation.json').read_text())
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
    cases=[];inputs={}
    for polar,color in zip(['0257','0258','0259','0260','0261'],colors):
        path=shock_dir/f'polar_{polar}.npz'
        record=next(c for c in shock_record['cases'] if c['polar']==polar)
        if sha256(path)!=record['npz_sha256']:raise ValueError(f'Modified extraction {polar}')
        fields=base/'working'/f'polar_{polar}'/'fields.npz'
        if sha256(fields)!=record['input_sha256']['fields.npz']:raise ValueError(f'Stale extraction {polar}')
        marker=next(c for c in separation_record['cases'] if c['polar']==polar)
        if marker['inputs']['fields.npz']!=record['input_sha256']['fields.npz']:raise ValueError('Inconsistent source fields')
        with np.load(path) as d,np.load(fields) as f:
            eta,xc=normalized_front(d['front_xyz_m'],f['coordinates'])
        cases.append(dict(name=f'PSP {record["angle_deg"]:.1f}',label=rf'$\alpha={record["angle_deg"]:.1f}^\circ$',eta=eta,xc=xc,
            separation=marker['span_fraction'],color=color,style='-',width=1.4))
        inputs[polar]={'shock':sha256(path),'fields':sha256(fields)}
    if sha256(numerical_dir/'fields.npz')!=numerical_record['output_sha256']:raise ValueError('Modified numerical data')
    with np.load(numerical_dir/'fields.npz') as d:
        eta,xc=normalized_front(d['front'],d['coordinates'])
    cases.insert(2,dict(name='AZDES 4.0',label=r'$\alpha=4.0^\circ$ (AZDES)',eta=eta,xc=xc,
        separation=numerical_record['separation_span_fraction'],color=colors[1],style='--',width=1.7))
    destination=args.output_dir or base/'paper/figures';destination.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({**STYLE,'text.usetex':not args.no_tex}):
        fig,ax=plt.subplots(figsize=(180/25.4,90/25.4))
        fig.subplots_adjust(left=.095,right=.98,top=.97,bottom=.16)
        for c in cases:
            valid=(c['eta']>=.1)&(c['eta']<=1)&np.isfinite(c['xc'])
            # Preserve any gaps rather than drawing through missing extractions.
            y=np.where(valid,c['xc'],np.nan)
            ax.plot(c['eta'],y,color=c['color'],ls=c['style'],lw=c['width'],label=c['label'])
            ax.axvline(c['separation'],color=c['color'],ls=':',lw=c['width'],alpha=.85)
            order=np.argsort(c['eta'][valid])
            at_marker=np.interp(c['separation'],c['eta'][valid][order],c['xc'][valid][order])
            ax.plot(c['separation'],at_marker,marker='x',color=c['color'],ms=5,mew=1.1)
        ax.set(xlim=(.1,1),ylim=(0,1),xlabel=r'$\eta$ [-]',
               ylabel=r'$x_s/c$ [-]')
        ax.set_xticks(np.arange(.1,1.01,.1));ax.set_yticks(np.arange(0,1.01,.2));ax.grid(alpha=.2,lw=.6)
        ax.legend(loc='upper right',ncol=3,frameon=True,framealpha=.95,edgecolor='0.85')
        stem=destination/'shock_front_comparison';export(fig,stem);plt.close(fig)
    metadata=dict(size_mm=[180,90],use_tex=not args.no_tex,
        span_definition='eta = (-y - min(-y_PSP))/(max(-y_PSP)-min(-y_PSP)); displayed PSP root-to-tip extent, not aircraft semispan',
        chord_definition='x/c = (x_front-x_LE)/(x_TE-x_LE) on each original/refined native row',
        displayed_span_min=.1,spline_rms_mm=shock_record['settings']['spline_rms_mm'],
        separation_definition='95%-local-chord pressure-gradient proxy; vertical lines show its span location, crosses show the corresponding position on the shock curve',
        input_hashes=inputs,shock_manifest_sha256=sha256(shock_dir/'extraction.json'),
        separation_manifest_sha256=sha256(separation_dir/'extraction.json'),
        numerical_manifest_sha256=sha256(numerical_dir/'preparation.json'),
        cases=[dict(name=c['name'],separation_span_fraction=c['separation']) for c in cases],script_sha256=sha256(Path(__file__)))
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(stem.with_suffix('.pdf'))


if __name__=='__main__':main()
