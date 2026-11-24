"""Compare phase-aligned leading SPOD modes on a portrait A4 page."""
import argparse
import json

import h5py
import hdf5plugin  # Register compressed HDF5 filters.
import matplotlib
import numpy as np
import torch
from dotenv import load_dotenv

from convert_surface import ROOT, configured_path
from mode_similarity_study import POLARS, prefix_basis, read_case
from plot_full_chord_mode_energy import CASES
from plot_band_mode_magnitude import chord_limits
from plot_style import STYLE, export

TARGETS = np.array([.10, .15, .20, .25, .40, .60])
SPEED_SUFFIX = '_no_sign_filter_no_amplitude_filter_slope_only_segments48_stmax0.7_minslope4'


def mean_along_shock_speed(output, key, frequencies):
    path = (output/'psp/working/displacement_directional' /
            f'polar_{key}_surface_band_speed{SPEED_SUFFIX}.npz')
    estimates = []
    with np.load(path) as data:
        edges = data['span_edges']
        # Use the outer half, as in the existing mean-speed comparison.
        span = edges[:-1] >= .5-1e-12
        widths = np.diff(edges)[span]
        for frequency in frequencies:
            index = int(np.argmin(abs(data['St'] - frequency)))
            speed = data['speed'][span, index]
            valid = np.isfinite(speed)
            assert valid.any(), (key, frequency)
            mean = float(np.average(speed[valid], weights=widths[valid]))
            outboard = float(np.average(speed[valid] > 0, weights=widths[valid]))
            direction = 'mixed'
            if abs(mean) >= .5:
                if mean > 0 and outboard >= .60:
                    direction = 'outboard'
                elif mean < 0 and outboard <= .40:
                    direction = 'inboard'
            estimates.append(dict(velocity_bin_St=float(data['St'][index]),
                                  mean_m_per_s=mean, outboard_fraction=outboard,
                                  direction=direction))
    return estimates


def aligned_real(modes, weights):
    moment = weights @ (modes**2)
    phase = .5*np.angle(moment)
    real = (modes*np.exp(-1j*phase)).real
    peak = np.argmax(np.abs(real), axis=0)
    sign = np.where(real[peak, np.arange(real.shape[1])] < 0, -1, 1)
    expected = .5*(weights @ np.abs(modes)**2 + np.abs(moment))
    np.testing.assert_allclose(weights @ real**2, expected, rtol=1e-12)
    limit = np.quantile(np.abs(modes), .95, axis=0)
    return real*sign/limit, phase + np.where(sign < 0, np.pi, 0), limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animate', action='store_true')
    args = parser.parse_args()
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize

    load_dotenv(ROOT/'.env')
    torch.set_num_threads(4)
    output = configured_path('OUTPUT_DIR')
    destination = output/'psp/exploratory/figures/a4_real_modes'
    working = output/'psp/working/a4_real_modes'
    for folder in (destination, working):
        folder.mkdir(parents=True, exist_ok=True)
    with np.load(output/'psp/working/polar_0258/fields.npz') as data:
        reference_span = -data['coordinates'][..., 1]/1000
    lower = reference_span.min() + .4*np.ptp(reference_span)
    upper = reference_span.max()
    norm = Normalize(-1, 1)
    records = {}
    animated_panels = []
    frames = 48
    if args.animate:
        from animate_psp_modes import reconstruct
        from matplotlib.animation import FFMpegWriter
    with plt.rc_context({**STYLE, 'axes.titlesize': 8}):
        fig, axes = plt.subplots(len(TARGETS), len(CASES), figsize=(210/25.4, 297/25.4))
        fig.subplots_adjust(left=.085, right=.97, top=.93, bottom=.12, wspace=.08, hspace=.10)
        direction_artists = []
        for col, key in enumerate(CASES):
            case = read_case(output, key)
            prefix_basis(case, working)
            bins = np.abs(case['St'][:, None] - TARGETS).argmin(axis=0)
            speeds = mean_along_shock_speed(output, key, case['St'][bins])
            modes = (case['basis'].to(torch.complex128) @ case['coeff'][:, bins]).numpy()
            if key == 'azdes':
                with h5py.File(output/'surface/geometry.h5') as geometry:
                    zone = geometry['zones/z000000']
                    xy = zone['points'][:, :2]
                    faces = zone['topology'][:]
                faces = faces[np.any(xy[faces, 1] >= lower, axis=1)]
                edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
                edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
                outline = xy[edges[counts == 1]]
                title = 'AZDES\n$\\alpha=4.0^\\circ$'
            else:
                with np.load(output/'psp/working'/f'polar_{key}'/'fields.npz') as data:
                    grid = data['coordinates']
                np.testing.assert_array_equal(grid.reshape(-1, 3)[case['indices']], case['coordinates'])
                xy = case['coordinates'][:, :2]*[.001, -.001]
                perimeter = np.concatenate((grid[0], grid[1:, -1], grid[-1, -2::-1], grid[-2:0:-1, 0]))
                perimeter = perimeter[:, :2]*[.001, -.001]
                outline = np.stack((perimeter, np.roll(perimeter, -1, axis=0)), axis=1)
                title = rf'PSP $\alpha={POLARS[key]:.1f}^\circ$'
            keep = (xy[:, 1] >= lower) & (xy[:, 1] <= upper)
            real, phase, limit = aligned_real(modes[keep], case['weight'].numpy()[keep])
            # Apply the visible-region alignment to all nodes before clipping the view.
            aligned = modes*np.exp(-1j*phase)/limit
            field = aligned.real
            np.testing.assert_allclose(field[keep], real, atol=1e-13)
            records[key] = dict(actual_St=case['St'][bins].tolist(), phase_rad=phase.tolist(),
                                propagation=speeds,
                                display_limits=limit.tolist(), source=case['source'])
            arrow_span = lower + np.array([.43, .65])*(upper-lower)
            arrow_points = [(chord_limits(outline, span)[0]-.025, span) for span in arrow_span]
            caps = [[(x, span) for x in chord_limits(outline, span)]
                    for span in (lower+1e-9, upper-1e-9)]
            for row, ax in enumerate(axes[:, col]):
                if key == 'azdes':
                    mesh = PolyCollection(xy[faces], array=field[faces, row].mean(axis=1),
                                          edgecolors='none', antialiased=False, snap=False,
                                          cmap='RdBu_r', norm=norm, rasterized=True)
                    ax.add_collection(mesh)
                    ax.text(.98, .04, f'{case["St"][bins[row]]:.3f}', ha='right', va='bottom',
                            transform=ax.transAxes, fontsize=7)
                else:
                    raster = np.full(grid.shape[:2], np.nan)
                    raster.flat[case['indices']] = field[:, row]
                    mesh = ax.pcolormesh(grid[..., 0]/1000, -grid[..., 1]/1000, raster,
                                         shading='nearest', cmap='RdBu_r', norm=norm, rasterized=True)
                ax.set(xlim=(.94, 1.31), ylim=(lower, upper), aspect='equal')
                ax.set_axis_off()
                if args.animate:
                    # Unit frequency represents cycle phase, independently of physical period.
                    sequence = reconstruct(torch.from_numpy(aligned[:, row]), 1., frames).numpy()
                    if key == 'azdes':
                        sequence = sequence[faces].mean(axis=1)
                    else:
                        full = np.full((grid.shape[0]*grid.shape[1], frames), np.nan)
                        full[case['indices']] = sequence
                        sequence = full.reshape(*grid.shape[:2], frames)
                    animated_panels.append((mesh, sequence))
                ax.add_collection(LineCollection(np.concatenate((outline, caps)),
                                                 colors='black', linewidths=.45, zorder=3))
                direction = speeds[row]['direction']
                start, end = arrow_points if direction != 'inboard' else arrow_points[::-1]
                artist = ax.annotate('', xy=end, xytext=start,
                                     arrowprops=dict(arrowstyle='<|-|>' if direction == 'mixed' else '-|>',
                                                     color='.2', lw=1))
                direction_artists.append(artist)
                if row == 0:
                    ax.set_title(title.replace('\n', ' '), pad=5)
                if col == 0:
                    ax.text(-.08, .5, rf'${TARGETS[row]:.2f}$', transform=ax.transAxes,
                            ha='right', va='center', fontsize=9)
        fig.text(.042, .943, r'$St$', ha='center', fontsize=9)
        color_axis = fig.add_axes((.30, .079, .40, .012))
        fig.colorbar(mesh, cax=color_axis, orientation='horizontal', extend='both',
                     label=r'$\mathrm{Re}(\phi_1 e^{-i\theta})/Q_{95}(|\phi_1|)$')
        stem = destination/'psp_azdes_leading_real_modes_a4_direction'
        if args.animate:
            writer = FFMpegWriter(fps=16, codec='h264',
                                  extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
            with writer.saving(fig, str(stem.with_suffix('.mp4')), dpi=165):
                for frame in range(frames):
                    for panel, sequence in animated_panels:
                        panel.set_array(sequence[..., frame])
                    writer.grab_frame()
            plt.close(fig)
            return
        export(fig, stem)
        for artist in direction_artists:
            artist.set_visible(False)
        export(fig, destination/'psp_azdes_leading_real_modes_a4')
        plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps({
        'target_St': TARGETS.tolist(), 'mode': 1, 'reference_span_limits_m': [lower, upper],
        'alignment': 'Maximize area-weighted squared real part over displayed span. '
                     'Choose the sign so the strongest absolute real value is positive.',
        'display': 'Each mode divided by its outer-span 95th percentile magnitude.',
        'cases': records,
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
