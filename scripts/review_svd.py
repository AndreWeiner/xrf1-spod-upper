"""Summarize full-surface rank choices and friction-component fluctuation norms."""
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT, configured_path
from plot_style import STYLE, WIDTH_MM, export


def main():
    load_dotenv(ROOT/'.env',override=False)
    output=configured_path('OUTPUT_DIR')/'analysis'
    base=output/'working/svd'
    records={name:json.loads((base/name/'run.json').read_text()) for name in ['cp','cf_xy','cf_xyz']}
    assert all(r['status']=='complete' for r in records.values())
    lines=['# SVD rank review','',
        'Area-weighted, temporally centered coefficient fluctuations; Float64; all 1,310 snapshots.','',
        '| State | 90% | 95% | 99% | 99.5% | 99.9% |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in records.items():
        values=r['ranks_for_energy_fraction']
        lines.append('| '+name+' | '+' | '.join(str(values[str(q)]) for q in [.9,.95,.99,.995,.999])+' |')
    lines += ['','## Common-rank candidates','',
        '| Rank | cp retained energy | cf_xy retained energy | cf_xyz retained energy |',
        '| --- | ---: | ---: | ---: |']
    for rank in [128,200,256,400,512,800]:
        values=[100*r['energy_fraction_at_rank'][str(rank)] for r in records.values()]
        lines.append(f'| {rank} | '+' | '.join(f'{v:.4f}%' for v in values)+' |')
    components=records['cf_xyz']['validation']['component_weighted_energies']
    total=sum(components.values())
    lines += ['','## Friction-component contributions','',
        'Fractions of the area-weighted, temporally centered three-component fluctuation norm:','']
    lines += [f'- `{key}`: {100*value/total:.4f}%.' for key,value in components.items()]
    lines += ['','These fractions describe the coefficient norm. They do not establish the influence of a component on any particular frequency or physical mechanism.','',
        'Rank remains a study choice. Equal ranks give equal reduced dimensions, not equal truncation errors. Freeze the chosen rank before comparing tapering.','',
        'The provisional adaptive cap is 50. Review cap-hit fractions and final residuals before deciding whether a larger cap is required.','']
    (base/'rank_review.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['single']/25.4,65/25.4),layout='constrained')
        labels=[r'$c_{fx}$',r'$c_{fy}$',r'$c_{fz}$']
        values=[100*v/total for v in components.values()]
        bars=ax.bar(labels,values,color=['#0072B2','#D55E00','#009E73'])
        ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set_ylim(0,min(100,max(values)*1.22));ax.set_ylabel(r'Weighted fluctuation norm (\%)')
        destination=output/'exploratory/figures/svd'
        destination.mkdir(parents=True,exist_ok=True)
        export(fig,destination/'cf_component_contributions');plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
