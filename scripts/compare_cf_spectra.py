"""Compare the two- and three-component friction spectra at the selected rank."""
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT,configured_path
from plot_style import STYLE,WIDTH_MM,export
from reference_scales import strouhal


def main():
    load_dotenv(ROOT/'.env',override=False)
    output=configured_path('OUTPUT_DIR')/'analysis'
    config=json.loads((ROOT/'config/spod_study.json').read_text())
    labels=[f'fixed_{k}' for k in config['fixed_tapers']]+[f'adaptive_{t:g}' for t in config['adaptive_tolerances']]
    rows=[];arrays={}
    for label in labels:
        with np.load(output/f'working/spod/cf_xy/{label}/spectra.npz') as xy,np.load(output/f'working/spod/cf_xyz/{label}/spectra.npz') as xyz:
            np.testing.assert_array_equal(xy['frequency'],xyz['frequency'])
            f=xy['frequency'];first=xy['eigvals'][:,0];second=xyz['eigvals'][:,0]
            ratio=np.divide(second-first,first,out=np.full_like(first,np.nan),where=first>0)
            peak=int(np.argmax(first[1:])+1)
            arrays[label+'_relative_change']=ratio;arrays[label+'_xy']=first;arrays[label+'_xyz']=second
            rows.append({'variant':label,'xy_peak_frequency_hz':float(f[peak]),'xy_peak_strouhal':float(strouhal(f[peak],'numerical')),'relative_change_at_xy_peak':float(ratio[peak]),
                'median_relative_change_positive_frequencies':float(np.nanmedian(ratio[1:]))})
    arrays['frequency_hz']=f
    arrays['strouhal']=strouhal(f,'numerical')
    np.savez(output/'working/spod/cf_component_spectral_comparison.npz',**arrays)
    record={'rank':400,'normalization':'same raw area quadrature and unit component factors',
        'scope':'Leading eigenvalues, not mode-shape or causal comparison',
        'limitation':'Adding cfz changes the truncated POD subspace; differences include both state augmentation and truncation effects.',
        'results':rows}
    (output/'working/spod/cf_component_spectral_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,1,figsize=(WIDTH_MM['double']/25.4,105/25.4),layout='constrained',sharex=True)
        for label,line,name,color in [('fixed_16','-','Fixed 16','#0072B2'),('adaptive_1e-05','--',r'Adaptive $10^{-5}$','#D55E00')]:
            valid=f>0;st=strouhal(f,'numerical')
            axs[0].loglog(st[valid],arrays[label+'_xy'][valid],color=color,ls=line,label=name+r', $(c_{fx},c_{fy})$')
            axs[0].loglog(st[valid],arrays[label+'_xyz'][valid],color=color,ls=':',label=name+r', $(c_{fx},c_{fy},c_{fz})$')
            axs[1].semilogx(st[valid],100*arrays[label+'_relative_change'][valid],color=color,ls=line,label=name)
        axs[0].set_ylabel(r'Leading eigenvalue $\lambda_1$');axs[0].legend(ncol=2,fontsize=8)
        axs[1].set_ylabel(r'Change in $\lambda_1$ [\%]');axs[1].set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$');axs[1].legend()
        axs[1].axhline(0,color='0.5',lw=.8)
        for ax in axs:ax.grid(alpha=.2)
        export(fig,output/'exploratory/figures/spod/cf_component_comparison');plt.close(fig)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
