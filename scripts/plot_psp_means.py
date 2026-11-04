"""Regenerate readable mean/FFT comparisons from saved PSP statistics only."""
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT,configured_path
from plot_style import STYLE,WIDTH_MM,export
from reference_scales import strouhal


def main():
    load_dotenv(ROOT/'.env',override=False)
    config=json.loads((ROOT/'config/psp_study.json').read_text())
    label=config['selected_mask'] or f'p{config["default_patch_size"]}_t{config["default_threshold"]:g}'
    output=configured_path('OUTPUT_DIR')/'psp'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count=0
    for directory in sorted((output/'working').glob('polar_*')):
        record=json.loads((directory/'run.json').read_text())
        if record['status']!='complete':continue
        selected=next(r for r in record['cases'] if r['label']==label)
        if selected['status']!='complete':continue
        destination=output/'exploratory/figures'/record['polar'];destination.mkdir(parents=True,exist_ok=True)
        with plt.rc_context({**STYLE,'text.usetex':config['use_tex']}):
            fig,axs=plt.subplots(2,1,figsize=(WIDTH_MM['double']/25.4,105/25.4),layout='constrained')
            for name,color in [('unmasked_finite','#0072B2'),(label,'#D55E00')]:
                title='Finite-domain reference' if name=='unmasked_finite' else ('Selected mask' if config['selected_mask'] else 'Provisional default mask')
                with np.load(directory/f'{name}.npz') as data:
                    axs[0].plot(data['time_seconds'],data['mean'],label=title,color=color,lw=.7)
                    st=strouhal(data['frequency_hz'],'psp',record['polar']);keep=st>0
                    axs[1].loglog(st[keep],data['hann_amplitude'][keep],label=title,color=color,lw=.9)
                    if name==label:
                        axs[1].loglog(st[keep],data['linearly_detrended_hann_amplitude'][keep],label='Linear trend removed',color='#009E73',ls='--',lw=.9)
            axs[0].set(xlabel='Time [s]',ylabel=r'$\langle C_p\rangle_A$')
            axs[1].set(xlabel=r'$St=f c_{\mathrm{MAC}}/U_\infty$',ylabel=r'Hann amplitude [$C_p$]')
            for ax in axs:ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
            export(fig,destination/'provisional_spatial_mean_spectrum');plt.close(fig)
        count+=1
    print(f'Regenerated spatial-mean/Fourier figures for {count} complete polars.',flush=True)


if __name__=='__main__':main()
