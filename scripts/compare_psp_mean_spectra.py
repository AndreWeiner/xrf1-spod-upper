"""Compare saved retained-area spatial-mean spectra across all PSP polars."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT,configured_path,sha256
from reference_scales import CONFIG as SCALE_CONFIG,strouhal
from plot_style import STYLE,WIDTH_MM,export


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-tex',action='store_true')
    args=parser.parse_args();load_dotenv(ROOT/'.env',override=False)
    config=json.loads((ROOT/'config/psp_study.json').read_text())
    mask=config['selected_mask'] or f'p{config["default_patch_size"]}_t{config["default_threshold"]:g}'
    output=configured_path('OUTPUT_DIR')/'psp'
    destination=output/'exploratory/figures';destination.mkdir(parents=True,exist_ok=True)
    curves=[]
    for directory in sorted((output/'working').glob('polar_*')):
        record=json.loads((directory/'run.json').read_text())
        if record['status']!='complete':raise ValueError(f'Incomplete polar {directory.name}')
        with np.load(directory/f'{mask}.npz') as data:
            st=strouhal(data['frequency_hz'],'psp',record['polar'])
            assert st.dtype==np.float64 and len(data['mean'])==record['snapshots']==3639
            curves.append({'polar':record['polar'],'angle':record['angle_of_attack_deg'],
                'st':st,'demeaned':data['hann_amplitude'].copy(),
                'detrended':data['linearly_detrended_hann_amplitude'].copy(),
                'source_sha256':sha256(directory/f'{mask}.npz')})
    if len(curves)!=8:raise ValueError('Expected all eight PSP polars')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#332288','#000000']
    styles=['-','--','-.',':','-','--','-.',':']
    style={**STYLE,'text.usetex':config['use_tex'] and not args.no_tex}
    for zoom,stem in [(False,'spatial_mean_spectra_all_polars'),(True,'spatial_mean_spectra_peak_zoom')]:
        with plt.rc_context(style):
            fig,axes=plt.subplots(2,1,figsize=(WIDTH_MM['double']/25.4,130/25.4),layout='constrained',sharex=True,sharey=True)
            values=[]
            for curve,color,line in zip(curves,colors,styles):
                valid=curve['st']>0
                if zoom:valid&=(curve['st']>=.24)&(curve['st']<=.31)
                label=curve['polar']+rf', $\alpha={curve["angle"]:g}^\circ$'
                for ax,key in zip(axes,['demeaned','detrended']):
                    ax.plot(curve['st'][valid],curve[key][valid],color=color,ls=line,lw=.85,label=label)
                    values.extend(curve[key][valid])
            for ax,title in zip(axes,['Temporal mean removed','Linear trend removed']):
                ax.set_yscale('log');ax.set_title(title);ax.set_ylabel(r'Hann amplitude [$C_p$]');ax.grid(alpha=.2)
                if not zoom:ax.set_xscale('log')
            if zoom:axes[-1].set_xlim(.24,.31)
            positive=np.asarray(values);positive=positive[positive>0]
            axes[0].set_ylim(positive.min()/1.3,positive.max()*1.5)
            axes[0].legend(ncol=4,fontsize=7,loc='lower left')
            axes[-1].set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$')
            export(fig,destination/stem);plt.close(fig)
    with plt.rc_context(style):
        fig,axes=plt.subplots(4,2,figsize=(WIDTH_MM['double']/25.4,180/25.4),layout='constrained',sharex=True,sharey=True)
        for ax,curve in zip(axes.flat,curves):
            valid=curve['st']>0
            ax.loglog(curve['st'][valid],curve['demeaned'][valid],color='#0072B2',lw=.7,label='Mean removed')
            ax.loglog(curve['st'][valid],curve['detrended'][valid],color='#D55E00',ls='--',lw=.7,label='Linear trend removed')
            ax.set_title(curve['polar']+rf', $\alpha={curve["angle"]:g}^\circ$');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        for ax in axes[:,0]:ax.set_ylabel(r'Amplitude [$C_p$]')
        for ax in axes[-1]:ax.set_xlabel(r'$St$')
        export(fig,destination/'spatial_mean_spectra_per_polar');plt.close(fig)
    metadata={'mask':mask,'provisional':config['selected_mask'] is None,'snapshots_per_polar':3639,
        'temporal_reduction':'max','spectrum':'One-sided, coherent-gain-corrected Hann amplitude; no zero padding',
        'amplitude_normalization':'None between polars; absolute Cp amplitudes',
        'reference_scales_sha256':sha256(SCALE_CONFIG),
        'domain_note':'Each polar uses its own fitted mask and retained-area normalization.',
        'inputs':[{'polar':c['polar'],'angle_deg':c['angle'],'source_sha256':c['source_sha256']} for c in curves]}
    (output/'working/spatial_mean_spectra_comparison.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Exported shared-axis, peak-zoom and per-polar comparisons from cached arrays.')

if __name__=='__main__':main()
