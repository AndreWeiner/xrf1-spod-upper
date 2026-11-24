"""Appendix comparison of full-record automatic PSP masks at alpha=4 degrees."""
import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path, sha256
from plot_style import STYLE, WIDTH_MM, export
from reference_scales import CONFIG as REFERENCE_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-tex', action='store_true')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    output = configured_path('OUTPUT_DIR')
    source = output / 'psp/working/polar_0258'
    record = json.loads((source / 'run.json').read_text())
    if (record['status'] != 'complete' or record['angle_of_attack_deg'] != 4
            or record['snapshots'] != 3639
            or record['config']['temporal_reduction'] != 'max'):
        raise ValueError('Expected completed full-record alpha=4 PSP masks')
    thresholds = [5, 10, 20]
    with np.load(source / 'masks.npz') as arrays:
        masks = [arrays[f'p5_t{t}'].copy() for t in thresholds]
        score = arrays['p5_score']
        extreme = arrays['p5_extreme_count'] > 0
    with np.load(source / 'fields.npz') as arrays:
        finite = arrays['finite_keep']
        coordinates = arrays['coordinates'].copy()
    if not np.isfinite(coordinates).all():
        raise ValueError('Nonfinite PSP coordinates')
    # Native PSP coordinates are millimetres; use the physical wing planform.
    reference = json.loads(REFERENCE_CONFIG.read_text())
    chord = reference['psp']['mean_aerodynamic_chord_m']
    x = coordinates[..., 0] / (1000 * chord)
    outboard = -coordinates[..., 1] / (1000 * chord)
    for threshold, mask in zip(thresholds, masks):
        if mask.dtype != bool or mask.shape != (465, 159):
            raise ValueError('Unexpected mask shape/type')
        np.testing.assert_array_equal(mask, finite & ~extreme & (score <= threshold))
    destination = args.output_dir or output / 'psp/paper/figures'
    destination.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    style = dict(STYLE, **{'text.usetex': not args.no_tex})
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM['double']/25.4, 85/25.4),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(left=.085, right=.995, bottom=.09, top=.88, wspace=.12)
        cmap = ListedColormap(['black', 'white'])
        for index, (ax, threshold, mask) in enumerate(zip(axes, thresholds, masks)):
            ax.pcolormesh(x, outboard, mask, shading='nearest',
                          cmap=cmap, vmin=0, vmax=1, rasterized=True)
            ax.set_aspect('equal')
            ax.set_facecolor('0.9')
            ax.set_title(rf'({chr(97+index)}) $T={threshold}$' + '\n'
                         + rf'${100*mask.mean():.1f}\%$ retained')
            ax.set_xticks([])
            ax.set_yticks([])
        # Reserve a separate bottom strip for the binary-mask key.
        fig.legend(handles=[Patch(facecolor='white', edgecolor='black', label='Retained'),
                            Patch(facecolor='black', label='Excluded'),
                            Patch(facecolor='0.9', label='Outside grid')],
                   loc='lower center', bbox_to_anchor=(.5, 0), ncol=3, frameon=False)
        stem = destination / 'psp_automatic_masks_alpha4'
        export(fig, stem)
        plt.close(fig)
    metadata = {
        'polar': '0258', 'angle_of_attack_deg': 4, 'patch_size': 5,
        'thresholds': thresholds, 'temporal_reduction': 'max', 'snapshots': 3639,
        'crop_pixels': 0, 'native_masks_applied': False,
        'orientation': 'physical planform: x=X/(1000*c_MAC), outboard=-Y/(1000*c_MAC); equal aspect; original coordinate origin',
        'rendering': 'nearest-cell pcolormesh on native coordinate grid; retained white, excluded black',
        'percentage_denominator': 'all 465 by 159 original pixels',
        'reference_chord_m': chord, 'reference_config_sha256': sha256(REFERENCE_CONFIG),
        'size_mm': [180, 85], 'use_tex': not args.no_tex,
        'retained_pixels': [int(m.sum()) for m in masks],
        'retained_fraction': [float(m.mean()) for m in masks],
        'input_sha256': {name: sha256(source/name) for name in ['run.json', 'masks.npz', 'fields.npz']},
        'script_sha256': sha256(Path(__file__)),
    }
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(stem.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
