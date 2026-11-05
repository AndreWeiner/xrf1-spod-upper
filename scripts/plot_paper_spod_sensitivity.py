"""Publication figures for cached PSP and numerical SPOD sensitivity results."""
import argparse
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT,configured_path,sha256
from reference_scales import strouhal,CONFIG as REFERENCE_CONFIG
from plot_style import STYLE,WIDTH_MM,export

RANKS=[200,400,800]
TOLERANCES=[1e-4,5e-5,1e-5]
TOLERANCE_LABELS=[r'$\varepsilon=10^{-4}$',r'$\varepsilon=5\times10^{-5}$',r'$\varepsilon=10^{-5}$']
BLUE='#0072B2';ORANGE='#D55E00'


def read_spectrum(path,*,rank,adaptive,tolerance=None,tapers=None):
    record=json.loads((path/'run.json').read_text())
    if record['status']!='complete' or record['rank']!=rank or record['adaptive']!=adaptive:
        raise ValueError(f'Incompatible spectrum: {path}')
    if tolerance is not None and record['tolerance']!=tolerance:raise ValueError('Tolerance mismatch')
    if tapers is not None and record['requested_max_tapers']!=tapers:raise ValueError('Taper count mismatch')
    digest=sha256(path/'spectra.npz')
    if digest!=record['spectra_sha256']:raise ValueError('Spectrum checksum mismatch')
    with np.load(path/'spectra.npz') as data:
        frequency=data['frequency'].copy();leading=data['eigvals'][:,0].copy()
    if frequency.dtype!=np.float64 or leading.dtype!=np.float64 or not np.isfinite(leading).all():raise ValueError('Invalid precision/data')
    return record,frequency,leading,{'run_sha256':sha256(path/'run.json'),'spectra_sha256':digest}


def psp_figure(output,style,destination=None):
    base=output/'psp';destination=destination or base/'paper/figures'
    destination.mkdir(parents=True,exist_ok=True);inputs={};curves={}
    # Each panel contains three actual saved spectra, without smoothing or normalization.
    for polar,angle in [('0258',4),('0261',6)]:
        selections={(r,5e-5) for r in RANKS}|{(200,t) for t in TOLERANCES}
        for rank,tol in sorted(selections):
            path=base/'working/spod'/f'polar_{polar}'/f'rank_{rank}'/f'adaptive_{tol:g}'
            record,frequency,leading,fingerprint=read_spectrum(path,rank=rank,adaptive=True,tolerance=tol,tapers=200)
            assert record['polar']==polar and record['angle_deg']==angle and record['nfft']==3639
            st=strouhal(frequency,'psp',polar);valid=st>0
            if not (leading[valid]>0).all():raise ValueError('Log plot requires positive leading eigenvalues')
            curves[polar,rank,tol]=(st[valid],leading[valid])
            inputs[str(path.relative_to(output))]=fingerprint
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,2,figsize=(WIDTH_MM['double']/25.4,120/25.4),sharex=True,sharey=True,layout='constrained')
        rank_styles=[('black','-'),(BLUE,'--'),(ORANGE,'-.')]
        tolerance_styles=[(BLUE,'--'),('black','-'),(ORANGE,'-.')]
        for column,(polar,angle) in enumerate([('0258',4),('0261',6)]):
            for rank,(color,line) in zip(RANKS,rank_styles):
                st,values=curves[polar,rank,5e-5]
                axes[0,column].loglog(st,values,color=color,ls=line,label=rf'$r={rank}$',zorder=3 if rank==200 else 2)
            for tol,label,(color,line) in zip(TOLERANCES,TOLERANCE_LABELS,tolerance_styles):
                st,values=curves[polar,200,tol]
                axes[1,column].loglog(st,values,color=color,ls=line,label=label,zorder=3 if tol==5e-5 else 2)
            axes[0,column].set_title(rf'({chr(97+column)}) $\alpha={angle}^\circ$, $\varepsilon=5\times10^{{-5}}$')
            axes[1,column].set_title(rf'({chr(99+column)}) $\alpha={angle}^\circ$, $r=200$')
            axes[1,column].set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$')
        for ax in axes.flat:
            ax.grid(alpha=.18,which='major');ax.legend(loc='lower left',fontsize=9,framealpha=.95,edgecolor='0.85')
        for ax in axes[:,0]:ax.set_ylabel(r'$\lambda_1$')
        all_x=np.concatenate([x for x,_ in curves.values()]);all_y=np.concatenate([y for _,y in curves.values()])
        axes[0,0].set_xlim(all_x.min(),all_x.max());axes[0,0].set_ylim(all_y.min()/1.18,all_y.max()*1.25)
        stem=destination/'psp_spod_sensitivity';export(fig,stem);plt.close(fig)
    metadata={'figure':'PSP leading-eigenvalue rank and tolerance sensitivity','size_mm':[180,120],
        'panels':{'a':{'polar':'0258','alpha_deg':4,'ranks':RANKS,'tolerance':5e-5},
                  'b':{'polar':'0261','alpha_deg':6,'ranks':RANKS,'tolerance':5e-5},
                  'c':{'polar':'0258','alpha_deg':4,'rank':200,'tolerances':TOLERANCES},
                  'd':{'polar':'0261','alpha_deg':6,'rank':200,'tolerances':TOLERANCES}},
        'maximum_tapers':200,'snapshot_count_per_polar':3639,
        'preprocessing':'snapshot weighted spatial mean and pointwise temporal mean removed; existing geometric nodal areas; automatic patch5 threshold10 max mask plus crop5 without Mask2',
        'axis_scales':'log-log; common limits in all panels',
        'normalization':'none','data_transform':'frequency converted to per-polar Strouhal; DC excluded; no smoothing/decimation',
        'reference_curve':'black solid: rank200, tolerance5e-5','rendering':'LaTeX' if style['text.usetex'] else 'mathtext',
        'source_sha256':sha256(Path(__file__)),'reference_scales_sha256':sha256(REFERENCE_CONFIG),'inputs':inputs}
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(stem.with_suffix('.pdf'))


def numerical_figure(output,style,destination=None):
    from numerical_cp_spod_sensitivity import result_path
    destination=destination or output/'analysis/paper/figures';destination.mkdir(parents=True,exist_ok=True)
    inputs={};curves={}
    selections={(r,5e-5) for r in RANKS}|{(200,t) for t in TOLERANCES}
    for rank,tol in sorted(selections):
        path=result_path(output,rank,tol)
        record,frequency,leading,fingerprint=read_spectrum(path,rank=rank,adaptive=True,tolerance=tol,tapers=200)
        assert record['nfft']==1310
        st=strouhal(frequency,'numerical');valid=st>0
        if not (leading[valid]>0).all():raise ValueError('Log plot requires positive leading eigenvalues')
        curves[rank,tol]=(st[valid],leading[valid]);inputs[str(path.relative_to(output))]=fingerprint
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(WIDTH_MM['double']/25.4,70/25.4),sharex=True,sharey=True,layout='constrained')
        for rank,(color,line) in zip(RANKS,[('black','-'),(BLUE,'--'),(ORANGE,'-.')]):
            st,values=curves[rank,5e-5]
            axes[0].loglog(st,values,color=color,ls=line,label=rf'$r={rank}$',zorder=3 if rank==200 else 2)
        for tol,label,(color,line) in zip(TOLERANCES,TOLERANCE_LABELS,[(BLUE,'--'),('black','-'),(ORANGE,'-.')]):
            st,values=curves[200,tol]
            axes[1].loglog(st,values,color=color,ls=line,label=label,zorder=3 if tol==5e-5 else 2)
        axes[0].set_title(r'(a) $\varepsilon=5\times10^{-5}$');axes[1].set_title(r'(b) $r=200$')
        axes[0].set_ylabel(r'$\lambda_1$')
        all_x=np.concatenate([x for x,_ in curves.values()]);all_y=np.concatenate([y for _,y in curves.values()])
        axes[0].set_xlim(all_x.min(),all_x.max());axes[0].set_ylim(all_y.min()/1.18,all_y.max()*1.25)
        for ax in axes:
            ax.set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$');ax.grid(alpha=.18,which='major')
            ax.legend(loc='lower left',fontsize=9,framealpha=.95,edgecolor='0.85')
        stem=destination/'numerical_cp_spod_sensitivity';export(fig,stem);plt.close(fig)
    metadata={'figure':'Numerical Cp leading-eigenvalue rank and tolerance sensitivity','size_mm':[180,70],
        'panels':{'a':{'ranks':RANKS,'tolerance':5e-5},'b':{'rank':200,'tolerances':TOLERANCES}},
        'maximum_tapers':200,'snapshot_count':1310,'preprocessing':'temporal mean removed; snapshot spatial mean retained; inherited geometric area weights',
        'axis_scales':'log-log; common limits','normalization':'none',
        'data_transform':'frequency converted to numerical Strouhal; DC excluded; no smoothing/decimation',
        'reference_curve':'black solid: rank200, tolerance5e-5 (shared panel reference, not a declaration of final numerical parameters)',
        'rendering':'LaTeX' if style['text.usetex'] else 'mathtext','source_sha256':sha256(Path(__file__)),
        'reference_scales_sha256':sha256(REFERENCE_CONFIG),'inputs':inputs}
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');print(stem.with_suffix('.pdf'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--figure',choices=['psp','numerical','all'],default='all')
    parser.add_argument('--no-tex',action='store_true');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    style={**STYLE,'text.usetex':not args.no_tex}
    output=configured_path('OUTPUT_DIR')
    if args.figure in ('psp','all'):psp_figure(output,style,args.output_dir)
    if args.figure in ('numerical','all'):numerical_figure(output,style,args.output_dir)

if __name__=='__main__':main()
