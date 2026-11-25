"""Paper Figure 2: temporal PSP mean/variance with translucent exclusion masks."""
import argparse
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path, sha256
from plot_style import STYLE, WIDTH_MM, export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--mask-alpha', type=float, default=.50)
    parser.add_argument('--no-tex', action='store_true')
    parser.add_argument('--shock-dir', type=Path, help='Saved output of extract_psp_mean_shocks.py')
    parser.add_argument('--separation-dir', type=Path)
    parser.add_argument('--numerical-dir', type=Path)
    args = parser.parse_args()
    if not 0 <= args.mask_alpha <= 1:
        parser.error('--mask-alpha must be between zero and one')
    load_dotenv(ROOT / '.env')
    source = configured_path('OUTPUT_DIR') / 'psp/working'
    polars = ['0257', '0258', '0259', '0260', '0261']
    shock_dir = args.shock_dir or configured_path('OUTPUT_DIR') / 'psp/paper/mean_shocks'
    extraction = json.loads((shock_dir/'extraction.json').read_text())
    separation_dir = args.separation_dir or configured_path('OUTPUT_DIR')/'psp/paper/trailing_edge_separation'
    separation = json.loads((separation_dir/'extraction.json').read_text())
    cases = []
    for polar in polars:
        directory = source / f'polar_{polar}'
        record = json.loads((directory / 'run.json').read_text())
        assert record['status'] == 'complete'
        shock_file = shock_dir / f'polar_{polar}.npz'
        prior = next(c for c in extraction['cases'] if c['polar'] == polar)
        if sha256(shock_file) != prior['npz_sha256']:
            raise ValueError(f'Shock extraction changed for {polar}')
        for name, digest in prior['input_sha256'].items():
            if sha256(directory/name) != digest:
                raise ValueError(f'Stale shock extraction for {polar}: {name}')
        with np.load(shock_file) as shock:
            front = shock['front_xyz_m'].copy()
        marker_record = next(c for c in separation['cases'] if c['polar'] == polar)
        for name, digest in marker_record['inputs'].items():
            if sha256(directory/name) != digest:
                raise ValueError(f'Stale trailing-edge extraction for {polar}')
        marker = np.asarray(marker_record['marker_xyz_m'])
        with np.load(directory / 'fields.npz') as fields, np.load(directory / 'masks.npz') as masks:
            keep = masks['p5_t10'].copy()
            keep[:5] = False; keep[-5:] = False
            keep[:, :5] = False; keep[:, -5:] = False
            cases.append(dict(polar=polar, angle=record['angle_of_attack_deg'],
                dataset='PSP', marker=marker, front=front, coordinates=fields['coordinates'].copy(), mean=fields['mean'].copy(),
                variance=fields['variance'].copy(), keep=keep,
                hashes={n: sha256(directory/n) for n in ['run.json','fields.npz','masks.npz']}))
    numerical_dir=args.numerical_dir or configured_path('OUTPUT_DIR')/'psp/paper/numerical_overview'
    numerical_record=json.loads((numerical_dir/'preparation.json').read_text())
    if numerical_record['status'] != 'complete' or sha256(numerical_dir/'fields.npz') != numerical_record['output_sha256']:
        raise ValueError('Incomplete or modified numerical preparation')
    with np.load(numerical_dir/'fields.npz') as numeric:
        numerical_case={name:numeric[name].copy() for name in ['coordinates','mean','variance','front','marker']}
    numerical_case.update(dataset='AZDES',polar='numerical_alpha4',angle=4.,keep=np.ones(numerical_case['mean'].shape,dtype=bool))
    cases.insert(2,numerical_case)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import Patch
    destination = args.output_dir or configured_path('OUTPUT_DIR')/'psp/paper/figures'
    destination.mkdir(parents=True, exist_ok=True)
    norms = [Normalize(-1.5, .1), Normalize(0, .03)]
    with plt.rc_context({**STYLE, 'text.usetex': not args.no_tex}):
        fig = plt.figure(figsize=(WIDTH_MM['double']/25.4, 295/25.4))
        # Overlap empty corners of successive planform bounding boxes so the
        # physical geometry fills an A4 text width without changing its aspect.
        axes = np.array([[fig.add_axes([.015 + col*.49, .738-row*.130, .475, .222],
                                      facecolor='none') for col in range(2)] for row in range(6)])
        for row, case in enumerate(cases):
            coords=case['coordinates']; span=-coords[...,1]/1000; chord=coords[...,0]/1000
            for col, name in enumerate(['mean','variance']):
                ax=axes[row,col]
                # Mirror the two planforms as in the original Figure 2.
                horizontal = -span if col == 0 else span
                field = np.ma.masked_invalid(case[name])
                artist=ax.pcolormesh(horizontal, -chord, field, shading='nearest',
                    cmap='RdBu_r' if col == 0 else 'magma', norm=norms[col], rasterized=True)
                overlay=np.ma.masked_where(case['keep'], np.ones(case['keep'].shape))
                if case['dataset'] == 'PSP':
                    ax.pcolormesh(horizontal, -chord, overlay, shading='nearest',
                        cmap=ListedColormap(['white']), vmin=0, vmax=1, alpha=args.mask_alpha, rasterized=True)
                eta=(span-span.min())/(span.max()-span.min())
                xi=(chord-chord[:,:1])/(chord[:,-1:]-chord[:,:1])
                grid_color='0.2' if col == 0 else '0.65'
                for reference in (eta,xi):
                    ax.contour(horizontal,-chord,reference,levels=[.25,.5,.75],
                               colors=[grid_color],linestyles='--',linewidths=.6,alpha=.8,zorder=3)
                ax.set_aspect('equal'); ax.set_anchor('E' if col == 0 else 'W'); ax.set_axis_off()
                if col == 0:
                    marker=case['marker']
                    ax.plot(marker[1], -marker[0], marker='x', color='black',
                            ms=4.5, mew=1.2, zorder=6)
                    front=case['front'].copy()
                    span_fraction = (-front[:, 1] - span.min()) / (span.max() - span.min())
                    front[span_fraction < .10] = np.nan
                    ax.plot(front[:, 1], -front[:, 0], color='white', lw=1.5, zorder=5)
                    tip = np.unravel_index(np.argmax(span), span.shape)
                    # Place the baseline just above the leading edge near the tip.
                    tip_x = float(horizontal[tip])
                    tip_y = float((-chord)[tip])
                    ax.text(tip_x + .025, tip_y + .085,
                            rf'$\alpha={case["angle"]:.1f}^\circ$ ({case["dataset"]})',
                            rotation=30, rotation_mode='anchor', ha='left', va='bottom', fontsize=14)
        for col,label in enumerate([r'$\mathrm{mean}(c_p)$ [-]',r'$\mathrm{var}(c_p)$ [-]']):
            cax=fig.add_axes([.065+col*.49,.078,.375,.008])
            cb=fig.colorbar(plt.cm.ScalarMappable(norm=norms[col],cmap='RdBu_r' if col==0 else 'magma'),cax=cax,orientation='horizontal',extend='both' if col==0 else 'max')
            cb.set_label(label)
            cb.set_ticks([-1.5, .1] if col == 0 else [0, .03])
            cb.set_ticklabels(['low', 'high'] if col == 0 else ['0', 'high'])
        fig.legend(handles=[Patch(facecolor='0.85',edgecolor='0.5',label='White overlay: excluded from analysis')],loc='lower center',bbox_to_anchor=(.5,0),frameon=False)
        stem=destination/'psp_moments_masked'
        export(fig,stem); plt.close(fig)
    metadata=dict(polars=polars,angles_deg=[c['angle'] for c in cases],
        definition='Unrepaired full-record temporal mean and variance; no spatial smoothing; native coordinate cells.',
        mask='Automatic p5_t10, full-record maximum reduction, plus five-pixel crop on all edges; no native Mask 2',
        mask_alpha=args.mask_alpha, mask_overlay_columns=['mean', 'variance'], size_mm=[180,295],use_tex=not args.no_tex,
        color_limits={'mean':[-1.5,.1],'variance':[0,.03]},
        excluded_fraction={c['polar']:float((~c['keep']).mean()) for c in cases},
        numerical_preparation_sha256=sha256(numerical_dir/'preparation.json'),
        row_datasets=[c['dataset'] for c in cases],
        reference_grid_fractions=[.25,.5,.75],
        separation_extraction_sha256=sha256(separation_dir/'extraction.json'),
        shock_display_min_span_fraction=.10,
        shock_extraction_sha256=sha256(shock_dir/'extraction.json'),
        inputs={c['polar']:c['hashes'] for c in cases if c['dataset']=='PSP'},script_sha256=sha256(Path(__file__)))
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(stem.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
