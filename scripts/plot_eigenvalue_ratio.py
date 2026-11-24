"""Six-case leading-to-second SPOD eigenvalue ratios for branch competition."""
from psp_settings import selected_settings, selected_result
import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT,configured_path,sha256
from plot_paper_mode_similarity import CASES
from reference_scales import strouhal
from plot_style import STYLE,WIDTH_MM,export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-tex',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    output=configured_path('OUTPUT_DIR')
    destination=args.output_dir or output/'paper/figures';destination.mkdir(parents=True,exist_ok=True)
    working=output/'analysis/working/eigenvalue_ratio';working.mkdir(parents=True,exist_ok=True)
    records=[];curves={}
    for key,label,angle in CASES:
        path=(output/'analysis/working/spod_rank_sensitivity/cp/rank_200/adaptive_0.0001' if key=='azdes' else
              selected_result(output, key))
        record=json.loads((path/'run.json').read_text())
        assert record['status']=='complete' and record['rank']==200 and record['adaptive']
        assert record['tolerance']==(1e-4 if key=='azdes' else selected_settings()[1])
        assert sha256(path/'spectra.npz')==record['spectra_sha256']
        with np.load(path/'spectra.npz') as arrays:
            st=strouhal(arrays['frequency'],'numerical' if key=='azdes' else 'psp',None if key=='azdes' else key)
            first=arrays['eigvals'][:,0].copy();second=arrays['eigvals'][:,1].copy()
            available=(arrays['n_tapers']>1)&(second>0)
        assert first.dtype==second.dtype==np.float64 and np.isfinite(first).all() and np.isfinite(second).all()
        ratio=np.divide(first,second,out=np.full_like(first,np.nan),where=available)
        assert np.all(ratio[available]>=1-1e-12)
        # Independent equality: ratio=1+relative gap, checked without rescaling spectra.
        np.testing.assert_allclose(ratio[available],1+(first[available]-second[available])/second[available],rtol=1e-14)
        np.savez(working/f'{key}.npz',St=st,lambda1=first,lambda2=second,ratio=ratio,available=available)
        curves[key]=(st,ratio)
        records.append({'case':key,'angle_deg':angle,'rank':200,'tolerance':record['tolerance'],
                        'spectra_sha256':record['spectra_sha256'],'unavailable_bins':int((~available).sum())})
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
    for upper,suffix in [(.7,''),(.35,'_zoom_01_035')]:
        extrema=[]
        for st,ratio in curves.values():
            visible=(st>=.1)&(st<=upper)&np.isfinite(ratio)
            extrema.extend(ratio[visible])
        with plt.rc_context(dict(STYLE,**{'text.usetex':not args.no_tex})):
            fig,axes=plt.subplots(2,3,sharex=True,sharey=True,
                                 figsize=(WIDTH_MM['double']/25.4,120/25.4),layout='constrained')
            for index,(ax,(key,label,angle)) in enumerate(zip(axes.flat,CASES)):
                st,ratio=curves[key]
                line,=ax.plot(st,ratio,color='C1' if key=='azdes' else 'C0',linewidth=1.,
                              linestyle='--' if key=='azdes' else '-')
                np.testing.assert_allclose(line.get_ydata(),ratio,equal_nan=True,rtol=0,atol=0)
                ax.axhline(1,color='0.35',linestyle=':',linewidth=.9)
                ax.set(xlim=(.1,upper),ylim=(.9,max(extrema)*1.15),yscale='log')
                ax.yaxis.set_major_locator(FixedLocator([1,2,5,10,20]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ticks=[.1,.3,.5,.7] if upper==.7 else [.1,.15,.2,.25,.3,.35]
                ax.set_xticks(ticks)
                ax.set_title(rf'({chr(97+index)}) {label}, $\alpha={angle:.1f}^\circ$',fontsize=9)
                ax.grid(alpha=.35,linestyle=':',which='major')
            fig.supxlabel(r'$St$',fontsize=10)
            fig.supylabel(r'$\lambda_1/\lambda_2$',fontsize=10)
            stem=destination/('spod_eigenvalue_ratio'+suffix);export(fig,stem);plt.close(fig)
        report={'status':'pass','inputs':records,'strouhal_limits':[.1,upper],
                'definition':'raw first eigenvalue divided by raw second eigenvalue; common amplitude normalization cancels',
                'display':'linear St, logarithmic ratio; shared limits; reference ratio=1; no smoothing; unavailable second modes masked',
                'size_mm':[180,120],'script_sha256':sha256(Path(__file__)),'use_tex':not args.no_tex}
        stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        print(stem.with_suffix('.pdf'),flush=True)


if __name__=='__main__':main()
