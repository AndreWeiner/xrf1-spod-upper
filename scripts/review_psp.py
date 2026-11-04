"""Summarize completed PSP polars without adopting a final mask."""
import json
from pathlib import Path
from dotenv import load_dotenv
from convert_surface import ROOT,configured_path
from plot_style import STYLE,WIDTH_MM,export
from reference_scales import strouhal


def summarize(output,config):
    records=[]
    for path in sorted((output/'working').glob('polar_*/run.json')):
        record=json.loads(path.read_text())
        if record['status']=='complete':records.append(record)
    if not records:return
    default=f'p{config["default_patch_size"]}_t{config["default_threshold"]:g}'
    rows=[]
    lines=['# PSP mask and spatial-mean review','',
        'All snapshots; automatic masks use temporal maximum reduction. The default mask remains provisional.','',
        '| Polar | Angle [deg] | Default retained area | Mean Cp | Spatial-mean std | Strongest candidate peak St |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for record in records:
        cases={row['label']:row for row in record['cases']};chosen=cases[default]
        if chosen['status']!='complete':continue
        peak=chosen['candidate_peak_frequencies_hz'][0] if chosen['candidate_peak_frequencies_hz'] else None
        row={'polar':record['polar'],'angle_deg':record['angle_of_attack_deg'],
             'snapshots':record['snapshots'],'default':chosen,'reference':cases['unmasked_finite'],
             'candidate_peak_strouhal':strouhal(chosen['candidate_peak_frequencies_hz'],'psp',record['polar']).tolist()}
        rows.append(row)
        peak_label=f'{float(strouhal(peak,"psp",record["polar"])):.6f}' if peak is not None else 'none'
        lines.append(f'| {record["polar"]} | {record["angle_of_attack_deg"]:g} | {100*chosen["retained_area_fraction"]:.2f}% | {chosen["mean_time_average"]:.6g} | {chosen["spatial_mean_temporal_std"]:.6g} | {peak_label} |')
    lines += ['','Spectral peaks are descriptive local maxima of the demeaned Hann amplitude spectrum, not statistical significance tests. Compare with the linearly detrended spectra to assess low-frequency trend leakage.','',
        'Each polar has its own mask and retained area. Differences between polars can include domain-selection effects; no common spatial mask is imposed.','']
    (output/'working/mask_review.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'working/mask_review.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=output/'exploratory/figures';destination.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(1,2,figsize=(WIDTH_MM['double']/25.4,65/25.4),layout='constrained')
        for row in rows:
            for ax,key in zip(axs,['mean_time_average','spatial_mean_temporal_std']):
                value=row['default'][key]
                ax.scatter(row['angle_deg'],value,s=20,color='#0072B2')
                ax.annotate(row['polar'],(row['angle_deg'],value),xytext=(3,4),textcoords='offset points',fontsize=7)
        axs[0].set_ylabel(r'Time mean of area-averaged $C_p$');axs[1].set_ylabel(r'Std of area-averaged $C_p$')
        for ax in axs:ax.set_xlabel('Angle of attack [deg]');ax.grid(alpha=.2)
        export(fig,destination/'polar_mean_summary');plt.close(fig)
    print(f'PSP review updated: {len(rows)} complete polars',flush=True)


def main():
    load_dotenv(ROOT/'.env',override=False)
    summarize(configured_path('OUTPUT_DIR')/'psp',json.loads((ROOT/'config/psp_study.json').read_text()))

if __name__=='__main__':main()
