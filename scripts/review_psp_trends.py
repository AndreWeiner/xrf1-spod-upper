"""Quantify preliminary spatial-mean trends using FlowTorch's native fit."""
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import torch
from flowtorch.analysis import linear_trend
from convert_surface import ROOT,configured_path
from plot_style import STYLE,WIDTH_MM,export
from reference_scales import strouhal


def main():
    load_dotenv(ROOT/'.env',override=False);torch.set_num_threads(4)
    config=json.loads((ROOT/'config/psp_study.json').read_text())
    label=config['selected_mask'] or f'p{config["default_patch_size"]}_t{config["default_threshold"]:g}'
    output=configured_path('OUTPUT_DIR')/'psp'
    rows=[];curves=[]
    for directory in sorted((output/'working').glob('polar_*')):
        r=json.loads((directory/'run.json').read_text())
        if r['status']!='complete':continue
        with np.load(directory/f'{label}.npz') as data:
            t=data['time_seconds'];mean=data['mean']
            fit=linear_trend(torch.from_numpy(mean)[None,:],time=t,accumulator_dtype=torch.float64)
            slope=float(fit.slope[0]);intercept=float(fit.intercept[0])
            with np.load(directory/'fields.npz') as fields,np.load(directory/'masks.npz') as masks:
                keep=masks[label];w=fields['area_weights'][keep]
                expected=np.average(fields['trend_slope'][keep],weights=w)
                np.testing.assert_allclose(slope,expected,rtol=1e-10,atol=1e-12)
            idx=int(np.argmax(data['linearly_detrended_hann_amplitude'][1:])+1)
            rows.append({'polar':r['polar'],'angle_deg':r['angle_of_attack_deg'],
                'mask':label,'provisional':config['selected_mask'] is None,
                'slope_cp_per_second':slope,'fitted_record_change_cp':slope*(t[-1]-t[0]),
                'r_squared':float(fit.r_squared[0]),'normalized_change':float(fit.normalized_change[0]),
                'strongest_detrended_bin_hz':float(data['frequency_hz'][idx]),
                'strongest_detrended_bin_strouhal':float(strouhal(data['frequency_hz'][idx],'psp',r['polar'])),
                'slope_commutation_absolute_error':abs(slope-expected)})
            curves.append((r['polar'],r['angle_of_attack_deg'],t.copy(),mean.copy(),intercept+slope*t))
    (output/'working/spatial_mean_trends.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Provisional spatial-mean trends','',
        'Native FlowTorch linear fits on the retained-area spatial mean; selected mask remains provisional unless explicitly adopted.','',
        '| Polar | Angle [deg] | Fitted change in Cp | Linear-fit R2 | Strongest detrended bin St |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f'| {r["polar"]} | {r["angle_deg"]:g} | {r["fitted_record_change_cp"]:.5g} | {r["r_squared"]:.3f} | {r["strongest_detrended_bin_strouhal"]:.6f} |')
    lines+=['','Record changes describe fitted linear drifts, not their cause. A dominant lowest-frequency bin does not resolve a distinct oscillation from record-scale trends. Candidate peaks are not significance tests.','']
    (output/'working/spatial_mean_trends.md').write_text('\n'.join(lines))
    if not curves:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(4,2,figsize=(WIDTH_MM['double']/25.4,155/25.4),layout='constrained')
        for ax,(polar,angle,t,mean,fit) in zip(axes.flat,curves):
            ax.plot(t,mean,color='#0072B2',lw=.5)
            ax.plot(t,fit,color='#D55E00',ls='--',lw=1.2)
            ax.set_title(f'{polar}, '+rf'$\alpha={angle:g}^\circ$')
            ax.set_ylabel(r'$\langle C_p\rangle_A$');ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(curves):]:ax.set_visible(False)
        for ax in axes[-1]:ax.set_xlabel('Time [s]')
        export(fig,output/'exploratory/figures/provisional_mean_trends');plt.close(fig)
    print(f'Native spatial-mean trend fits and commutation checks passed for {len(rows)} polars.')

if __name__=='__main__':main()
