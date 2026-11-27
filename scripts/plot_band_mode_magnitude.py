"""Compare band-integrated leading SPOD contributions on the native wing grids."""
import argparse
import json

import h5py
import hdf5plugin  # Registers the compressed geometry filters.
import matplotlib
import numpy as np
import torch
from dotenv import load_dotenv

from convert_surface import ROOT, configured_path, sha256
from mode_similarity_study import POLARS, prefix_basis, read_case
from plot_full_chord_mode_energy import CASES
from plot_style import STYLE, export
from scipy.interpolate import LinearNDInterpolator

BAND = (0.25, 0.70)


def replace_masked_values(field, ipsp_mask, x_ipsp, y_ipsp, method=LinearNDInterpolator):
    """Replace masked values in 2D PSP data.

    field - 2D field to which apply the mask
    ipsp_mask - 2D boolean mask indicating bad values (zero/False)
    x/_ipsp - geometric pixel coordinates
    method - interpolation method used to replace masked values
    """
    mask = ipsp_mask.type(torch.bool).flatten()
    points_clean = torch.column_stack([x_ipsp.flatten()[mask], y_ipsp.flatten()[mask]])
    interp = method(points_clean, field.flatten()[mask], fill_value=field.median())
    fill_values = torch.from_numpy(interp(x_ipsp.flatten()[~mask], y_ipsp.flatten()[~mask]))
    field_clean = field.clone().flatten()
    field_clean[~mask] = fill_values.type(field.dtype)
    return field_clean.reshape(field.shape)


def compute(output, destination, key, eigenvalue_weighted):
    case = read_case(output, key)
    prefix_basis(case, destination)
    st = case['St']
    edges = np.r_[st[0] - (st[1] - st[0])/2,
                  (st[:-1] + st[1:])/2, st[-1] + (st[-1] - st[-2])/2]
    widths = np.diff(np.clip(edges, *BAND))
    if eigenvalue_weighted:
        selected = np.flatnonzero(widths > 0)
        fractions = widths[selected] / np.diff(edges)[selected]
        weights = torch.from_numpy(fractions * case['leading_eigenvalues'][selected])
    else:
        selected = np.flatnonzero((st >= BAND[0]) & (st <= BAND[1]) & (widths > 0))
        weights = torch.from_numpy(widths[selected] / widths[selected].sum())
    basis = case['basis'].to(torch.complex128)
    magnitude = torch.zeros(basis.shape[0], dtype=torch.float64)
    for bins, window_weights in zip(np.array_split(selected, np.arange(32, len(selected), 32)),
                                     weights.split(32)):
        modes = basis @ case['coeff'][:, bins]
        amplitude = modes.abs()
        if eigenvalue_weighted:
            amplitude.square_()
        magnitude += amplitude @ window_weights
    if eigenvalue_weighted:
        torch.testing.assert_close(case['weight'] @ magnitude, weights.sum())
        magnitude.sqrt_()
    elif key != 'azdes':
        # Convert the unit-area-normalized shape from mm^-1 to m^-1.
        magnitude *= 1000
    values = magnitude.numpy()
    assert np.isfinite(values).all() and values.max() > 0
    return values, case, selected.size


def separation_markers(output):
    paper = output/'psp/paper'
    psp = json.loads((paper/'trailing_edge_separation/extraction.json').read_text())
    numerical = json.loads((paper/'numerical_overview/preparation.json').read_text())
    assert numerical['status'] == 'complete'
    assert sha256(output/'surface/geometry.h5') == numerical['source_geometry_sha256']
    numerical_fields = paper/'numerical_overview/fields.npz'
    assert sha256(numerical_fields) == numerical['output_sha256']
    with np.load(numerical_fields) as data, h5py.File(output/'surface/geometry.h5') as geometry:
        # Undo the chordwise translation used to map AZDES to the PSP grid.
        marker = data['marker'].copy()
        marker[0] += geometry['zones/z000000/points'][:, 0].min() - data['coordinates'][..., 0].min()/1000
    markers = {'azdes': marker.tolist()}
    for record in psp['cases']:
        key = record['polar']
        if key in CASES:
            directory = output/'psp/working'/f'polar_{key}'
            for name, digest in record['inputs'].items():
                assert sha256(directory/name) == digest
            markers[key] = record['marker_xyz_m']
    return markers


def chord_limits(segments, span):
    """Intersect the projected wing boundary with a constant-span line."""
    start, end = segments.transpose(1, 0, 2)
    crosses = (start[:, 1] > span) != (end[:, 1] > span)
    start, end = start[crosses], end[crosses]
    fraction = (span - start[:, 1]) / (end[:, 1] - start[:, 1])
    x = start[:, 0] + fraction * (end[:, 0] - start[:, 0])
    return float(x.min()), float(x.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-color', action='store_true')
    parser.add_argument('--eigenvalue-weighted', action='store_true',
                        help='Plot leading-mode band RMS with mean-pressure separation markers')
    parser.add_argument('--inpaint-masked', action='store_true',
                        help='Replace masked PSP pixels by interpolation from retained pixels')
    args = parser.parse_args()
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import Stroke, Normal

    load_dotenv(ROOT/'.env')
    torch.set_num_threads(4)
    output = configured_path('OUTPUT_DIR')
    destination = output/'psp/exploratory/figures/band_mode_magnitude'
    paper_destination = output/'psp/paper/figures/band_mode_magnitude'
    destination.mkdir(parents=True, exist_ok=True)
    paper_destination.mkdir(parents=True, exist_ok=True)
    working = output/'psp/working/band_mode_magnitude'
    working.mkdir(parents=True, exist_ok=True)
    panels = {}
    records = {}
    chords = {}
    markers = separation_markers(output)
    for key in CASES:
        values, case, count = compute(output, working, key, args.eigenvalue_weighted)
        if key == 'azdes':
            with h5py.File(output/'surface/geometry.h5') as geometry:
                zone = geometry['zones/z000000']
                xy = zone['points'][:, :2]
                faces = zone['topology'][:]
            assert len(xy) == len(values)
            panels[key] = (xy[faces], values[faces].mean(axis=1))
            edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
            edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
            chords[key] = chord_limits(xy[edges[counts == 1]], -markers[key][1])
        else:
            with np.load(output/'psp/working'/f'polar_{key}'/'fields.npz') as data:
                grid = data['coordinates']
            np.testing.assert_array_equal(grid.reshape(-1, 3)[case['indices']], case['coordinates'])
            field = np.full(grid.shape[:2], np.nan)
            field.flat[case['indices']] = values
            if args.inpaint_masked:
                keep = torch.zeros(grid.shape[:2], dtype=torch.bool)
                keep.view(-1)[torch.from_numpy(case['indices'])] = True
                field_t = torch.from_numpy(np.nan_to_num(field, nan=0.))
                field = replace_masked_values(field_t, keep,
                    torch.from_numpy(grid[..., 0]/1000), torch.from_numpy(-grid[..., 1]/1000)).numpy()
                assert np.isfinite(field).all()
            panels[key] = (grid[..., 0]/1000, -grid[..., 1]/1000, field)
            outline = np.concatenate((grid[0], grid[1:, -1], grid[-1, -2::-1], grid[-2:0:-1, 0]))
            outline = outline[:, :2] * np.array([.001, -.001])
            segments = np.stack((outline, np.roll(outline, -1, axis=0)), axis=1)
            chords[key] = chord_limits(segments, -markers[key][1])
        records[key] = {'bins': int(count), 'source': case['source'],
                        'separation_marker_xyz_m': markers[key],
                        'separation_line_x_limits_m': chords[key]}
        print(key, count, 'bins', flush=True)

    pooled = np.concatenate([panel[-1][np.isfinite(panel[-1])] for panel in panels.values()])
    limit = float(np.quantile(pooled, .995))
    norm = LogNorm(limit/100, limit) if args.log_color else Normalize(0, limit)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(8.3, 6.2), sharex=True, sharey=True,
                                 layout='constrained')
        for ax, key in zip(axes.flat, CASES):
            panel = panels[key]
            if key == 'azdes':
                mesh = PolyCollection(panel[0], array=panel[1], edgecolors='none',
                                      cmap='viridis', norm=norm, rasterized=True,
                                      antialiased=False, snap=False)
                ax.add_collection(mesh)
                label = r'AZDES $\alpha=4.0^\circ$'
            else:
                mesh = ax.pcolormesh(*panel, shading='nearest', cmap='viridis',
                                     norm=norm, rasterized=True)
                label = rf'PSP $\alpha={POLARS[key]:.1f}^\circ$'
            ax.set(title=label, aspect='equal', xlim=(.74, 1.31), ylim=(.06, .80),
                   xticks=[], yticks=[])
            span = -markers[key][1]
            ax.plot(chords[key], [span, span], color='white', lw=1.2,
                    path_effects=[Stroke(linewidth=2.5, foreground='black'), Normal()])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if args.eigenvalue_weighted:
            title = 'Leading-mode RMS'
            color_label = r'Leading-mode $c_p$ RMS [-]'
            quantity = 'leading_mode_rms'
        else:
            title = 'Leading SPOD mode magnitude'
            color_label = r'Band-averaged $|\phi_1|$ [$\mathrm{m}^{-1}$]'
            quantity = 'leading_mode_magnitude'
        fig.legend(handles=[Line2D([], [], color='white', lw=1.2,
                                   path_effects=[Stroke(linewidth=2.5, foreground='black'), Normal()],
                                   label='Spanwise position of the trailing edge separation')],
                   loc='outside lower center', frameon=False)
        suptitle_text = fig.suptitle(title + r', $0.25 \leq St \leq 0.70$')
        fig.colorbar(mesh, ax=axes, extend='both' if args.log_color else 'max',
                     label=color_label)
        suffix = '_log' if args.log_color else ''
        stem = destination/f'psp_azdes_{quantity}_025_070{suffix}'
        export(fig, stem)
        # Paper variant: same figure without the title and with tighter panel spacing.
        suptitle_text.set_visible(False)
        fig.get_layout_engine().set(w_pad=0, h_pad=0, wspace=0, hspace=0)
        paper_stem = paper_destination/stem.name
        export(fig, paper_stem)
        plt.close(fig)
    def report(stem):
        stem.with_suffix('.json').write_text(json.dumps({
        'band_St': BAND, 'mode': 1,
        'method': (
            'Square root of the sum of one-sided leading eigenvalue times squared mode magnitude, '
            'weighted by the fraction of each native bin inside the band.'
            if args.eigenvalue_weighted else
            'Bin-width-weighted mean of absolute unit area-norm modes. No eigenvalue weighting.'
        ),
        'color_limits': [norm.vmin, norm.vmax],
        'color_scale': 'logarithmic' if args.log_color else 'linear', 'cases': records,
        'masked_value_handling': ('masked pixels replaced by linear interpolation from retained pixels; '
                                  'median fill outside the retained convex hull') if args.inpaint_masked else 'none',
    }, indent=2)+'\n')
    report(stem);report(paper_stem)


if __name__ == '__main__':
    main()
