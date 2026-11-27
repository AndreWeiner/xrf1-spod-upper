"""Reference-band normalized five-angle PSP and numerical Cp SPOD overview using FlowTorch."""
from psp_settings import selected_settings, selected_result
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.visualization import plot_spod_spectrum
from convert_surface import ROOT, configured_path, sha256
from plot_style import STYLE, WIDTH_MM, export
from reference_scales import reference, frequency_hz, CONFIG as REFERENCE_CONFIG



def smooth_log_frequency(coordinate, values, sigma_decades=.02):
    """Smooth linear eigenvalue amplitudes on a uniform log10-frequency grid."""
    logarithm = np.log10(coordinate)
    grid = np.linspace(logarithm[0], logarithm[-1], 4096)
    spacing = grid[1]-grid[0]
    radius = int(np.ceil(4*sigma_decades/spacing))
    offset = np.arange(-radius, radius+1)*spacing
    kernel = np.exp(-.5*(offset/sigma_decades)**2)
    kernel /= kernel.sum()
    interpolated = np.interp(grid, logarithm, values)
    filtered = np.convolve(np.pad(interpolated, radius, mode='reflect'), kernel, mode='valid')
    return np.interp(logarithm, grid, filtered)

def read_result(path, rank, tolerance):
    record = json.loads((path/'run.json').read_text())
    if (record['status'] != 'complete' or record['rank'] != rank
            or not record['adaptive'] or record['tolerance'] != tolerance
            or record['requested_max_tapers'] != 200):
        raise ValueError(f'Unexpected SPOD settings: {path}')
    if sha256(path/'spectra.npz') != record['spectra_sha256']:
        raise ValueError(f'Spectrum checksum mismatch: {path}')
    with np.load(path/'spectra.npz') as arrays:
        f = arrays['frequency'].copy()
        values = arrays['eigvals'].copy()
        bandwidth = arrays['half_bandwidth'].copy()
    if (f.dtype != np.float64 or values.dtype != np.float64
            or not np.isfinite(values).all() or not np.isfinite(f).all()
            or values.shape != (len(f), min(rank, record['effective_max_tapers'])) or np.min(values) < 0
            or not values.sum() > 0):
        raise ValueError(f'Invalid eigenvalue data: {path}')
    if bandwidth.shape != f.shape or not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0):
        raise ValueError(f'Invalid adaptive half-bandwidth: {path}')
    return record, f, values, bandwidth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-tex', action='store_true')
    args = parser.parse_args()
    load_dotenv(ROOT/'.env')
    output = configured_path('OUTPUT_DIR')
    # Adjacent PSP/numerical 4-degree panels on the first row.
    selections = [('0257', 3.5), ('0258', 4), ('numerical', 4),
                  ('0259', 4.5), ('0260', 5), ('0261', 6)]
    psp_snapshots = {polar: json.loads((output/'psp/working/svd'/f'polar_{polar}'/'run.json').read_text())['snapshots']
                     for polar, _ in selections if polar != 'numerical'}
    numerical_snapshots = json.loads((output/'analysis/working/svd/cp/run.json').read_text())['snapshots']
    frequencies, eigenvalues, timescales, titles, metadata, bandwidths = [], [], [], [], [], []
    for index, (polar, angle) in enumerate(selections):
        numerical = polar == 'numerical'
        rank, tolerance = (200, 1e-4) if numerical else selected_settings()
        path = (output/'analysis/working/spod_rank_sensitivity/cp/rank_200/adaptive_0.0001' if numerical else
                selected_result(output, polar))
        record, f, values, bandwidth = read_result(path, rank, tolerance)
        if not numerical:
            assert record['polar'] == polar and record['angle_deg'] == angle
            assert record['snapshots'] == psp_snapshots[polar]
            assert record['subtract_snapshot_spatial_mean'] and record['subtract_pointwise_temporal_mean']
            assert record['domain']['automatic_mask'] == 'p5_t10'
            assert record['domain']['crop_pixels_each_edge'] == 5 and record['domain']['native_mask'] is None
        else:
            assert record['nfft'] == numerical_snapshots
        dataset = 'numerical' if numerical else 'psp'
        chord, velocity = reference(dataset, None if numerical else polar)
        frequencies.append(torch.from_numpy(frequency_hz(f, dataset)))
        eigenvalues.append(torch.from_numpy(values))
        bandwidths.append(torch.from_numpy(frequency_hz(bandwidth, dataset)))
        timescales.append(chord/velocity)
        titles.append(rf'({chr(97+index)}) '+('AZDES' if numerical else 'PSP')
                      + rf', $\alpha={angle:g}^\circ$')
        metadata.append({'dataset': dataset, 'polar': None if numerical else polar,
                         'angle_deg': angle, 'rank': rank, 'tolerance': tolerance,
                         'total_energy': float(values.sum()), 'frequency_bins': len(f),
                         'eigenvalues_per_frequency': values.shape[1],
                         'run_sha256': sha256(path/'run.json'),
                         'spectra_sha256': sha256(path/'spectra.npz'),
                         'cap_hit_fraction': record['cap_hit_fraction'],
                         'tolerance_met_fraction': record['tolerance_met_fraction']})
    coordinates = [f.numpy()*t for f, t in zip(frequencies, timescales)]
    x_upper = min(float(x.max()) for i, x in enumerate(coordinates) if i != 2)
    peak_references = {}
    for name, index in [('psp_alpha4', 1), ('azdes', 2)]:
        in_band = (coordinates[index] >= .2) & (coordinates[index] <= x_upper)
        candidates = np.flatnonzero(in_band)
        if not candidates.size:
            raise ValueError(f'Empty peak normalization interval: {name}')
        leading_raw = eigenvalues[index].numpy()[:, 0]
        peak_index = candidates[np.argmax(leading_raw[candidates])]
        peak = float(leading_raw[peak_index])
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(f'Invalid reference peak: {name}')
        peak_references[name] = {'maximum_leading_eigenvalue': peak,
                                 'peak_St': float(coordinates[index][peak_index]),
                                 'bin_count': int(candidates.size)}
        np.testing.assert_allclose((leading_raw[in_band]/peak).max(), 1., rtol=1e-14)
    peak_factors = [peak_references['azdes' if i == 2 else 'psp_alpha4']['maximum_leading_eigenvalue']
                    for i in range(len(selections))]
    factors = peak_factors
    plotted_eigenvalues = [values/factor for values, factor in zip(eigenvalues, factors)]
    for item, factor in zip(metadata, factors):
        item['normalization_factor'] = factor
    destination = args.output_dir or output/'paper/figures'
    destination.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    style = dict(STYLE, **{'text.usetex': not args.no_tex})
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,
                                 figsize=(WIDTH_MM['double']/25.4, 120/25.4),
                                 layout='constrained')
        for ax, f, values, timescale, bandwidth, title, item in zip(
                axes.flat, frequencies, plotted_eigenvalues, timescales, bandwidths, titles, metadata):
            coordinate = f.numpy()*timescale
            targets = np.geomspace(.05*1.4, x_upper/1.4, 5)
            indices = np.unique([np.abs(coordinate-target).argmin() for target in targets])
            plot_spod_spectrum(f, values, n_show=5, show_sum=True,
                               normalize_energy=False, energy_density=False,
                               reference_timescale=timescale, half_bandwidth=bandwidth,
                               ax=ax, show_legend=False, mode_color='C0', sum_color='black',
                               min_opacity=.25, mode_kwargs={'linewidth': 1.1},
                               sum_kwargs={'linewidth': 1.4},
                               bandwidth_kwargs={'errorevery': indices.tolist(), 'alpha': .5,
                                                 'elinewidth': .9, 'capsize': 2})
            ax.axvspan(.2, .7, color='0.5', alpha=.12, linewidth=0, zorder=-10)
            ax.set_title(title)
            ax.set_xlabel('')
            ax.set_ylabel('')
            item['resolution_markers'] = [
                {'St': float(coordinate[i]), 'half_bandwidth_St': float(bandwidth[i]*timescale)}
                for i in indices]
            # Verify native error-bar endpoints against saved adaptive half-bandwidths.
            segments = ax.collections[0].get_segments()
            assert len(segments) == len(indices)
            for segment, i in zip(segments, indices):
                width = float(bandwidth[i]*timescale)
                np.testing.assert_allclose(segment[:, 0], [coordinate[i]-width, coordinate[i]+width], rtol=1e-13, atol=1e-15)
        positive_y = []
        # Independently verify the native plotted arrays against the saved spectra.
        for ax, f, values, timescale, factor in zip(axes.flat, frequencies, eigenvalues, timescales, factors):
            expected = [values.numpy().sum(1)/factor] + [values.numpy()[:, i]/factor for i in range(5)]
            assert len(ax.lines) == 8
            for line, y in zip(ax.lines[:6], expected):
                np.testing.assert_allclose(line.get_xdata(), f.numpy()*timescale, rtol=1e-14)
                actual = np.asarray(line.get_ydata())
                valid = y > np.finfo(np.float64).tiny
                np.testing.assert_allclose(actual[valid], y[valid], rtol=2e-14)
                coordinate = f.numpy()*timescale
                visible = (coordinate >= .05) & (coordinate <= x_upper)
                positive_y.extend(y[valid & visible])
            # DC is omitted on the logarithmic frequency axis.
            for line in ax.lines[:6]:
                y = np.array(line.get_ydata(), copy=True)
                y[f.numpy() == 0] = np.nan
                line.set_ydata(y)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.,)))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.4, which='major', linestyle=':', linewidth=.8)
            for st in [.2, .3, .4, .5, .6, .7]:
                ax.axvline(st, color=plt.rcParams['grid.color'],
                           linestyle=':',
                           linewidth=.8, alpha=.4, zorder=0.5)
        axes.flat[0].set_xlim(.05, x_upper)
        axes.flat[0].set_ylim(min(positive_y)/1.3, max(positive_y)*1.3)
        fig.supxlabel(r'$St$ [-]', fontsize=STYLE['axes.labelsize'])
        fig.supylabel(r'$\lambda/\lambda_{\mathrm{ref}}$ [-]', fontsize=STYLE['axes.labelsize'])
        handles = list(axes.flat[0].lines[:2]) + [axes.flat[0].containers[0]]
        # Legend in one panel avoids taking space from the shared labels.
        axes.flat[0].legend(handles, [r'$\sum_i\lambda_i$', r'$\lambda_1,\ldots,\lambda_5$', r'$\Delta St_{1/2}$'],
                            loc='lower left', fontsize=8)
        stem = destination/'spod_spectra_overview'
        export(fig, stem)
        plt.close(fig)
    report = {'status': 'pass', 'panels': metadata, 'size_mm': [180, 120],
              'normalization': 'Divide all eigenvalues by raw leading-eigenvalue maximum at St >= 0.2 within displayed range. AZDES uses its own alpha4 reference; all PSP panels share primary PSP alpha4 reference. Same factors as the two-panel leading-eigenvalue figure; no density conversion.',
              'reference_shading_St': [0.2, 0.7],
              'reference_grid_lines_St': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
              'resolution_markers': 'Five approximately log-spaced samples per panel; native FlowTorch horizontal error bars show plus/minus adaptive half-bandwidth in St on summed spectrum',
              'normalization_band_St': [.2, x_upper],
              'normalization_references': peak_references,
              'strouhal_limits': [.05, x_upper],
              'limit_definition': 'fixed 0.05 lower St limit to minimum of PSP maximum Strouhal values; y limits from visible positive curve values with 30 percent multiplicative padding',
              'display': 'first five eigenvalues and sum over ALL eigenvalues; log-log; common AZDES-to-PSP frequency interval; DC hidden; no smoothing',
              'comparison_limitations': 'Different temporal records/frequency-bin widths; reference-scaled bin energies, not spectral densities; separate PSP and AZDES scales remove absolute cross-dataset amplitude information while preserving PSP angle-to-angle levels. PSP subtracts snapshot spatial mean; numerical does not.',
              'curve_validation': 'All 36 native curves match saved eigenvalues and their sums divided by raw alpha4 leading-eigenvalue peaks; both reference leading peaks equal one in the search interval',
              'rendering': 'mathtext' if args.no_tex else 'LaTeX',
              'reference_scales_sha256': sha256(REFERENCE_CONFIG),
              'flowtorch_plot_sha256': sha256(Path(inspect.getfile(inspect.unwrap(plot_spod_spectrum)))),
              'script_sha256': sha256(Path(__file__))}
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(stem.with_suffix('.pdf'))
    with plt.rc_context(style):
        fig, single_axis = plt.subplots(figsize=(WIDTH_MM['double']/25.4, 85/25.4),
                                       layout='constrained')
        axes = [single_axis]
        visible_values = []
        panel_indices = [[0, 1, 3, 4, 5, 2]]
        from matplotlib.ticker import FuncFormatter
        for panel, (ax, indices) in enumerate(zip(axes, panel_indices)):
            ax.axvspan(.2, .7, color='0.5', alpha=.12, linewidth=0, zorder=-10)
            for index in indices:
                polar, angle = selections[index]
                coordinate, values, factor = coordinates[index], eigenvalues[index], peak_factors[index]
                leading = values.numpy()[:, 0]/factor
                positive = coordinate > 0
                raw_leading = leading.copy()
                if polar != 'numerical':
                    ax.plot(coordinate[positive], raw_leading[positive],
                            color=f'C{index}', linewidth=.55, alpha=.3, zorder=1,
                            label='_nolegend_')
                    leading[positive] = smooth_log_frequency(coordinate[positive], leading[positive])
                else:
                    np.testing.assert_array_equal(leading, raw_leading)
                if not np.isfinite(leading).all() or np.any(leading < 0):
                    raise ValueError('Invalid eigenvalue curve')
                keep = (coordinate >= .05) & (coordinate <= x_upper) & (leading > 0)
                visible_values.extend(leading[keep])
                label = ('AZDES' if polar == 'numerical' else 'PSP') + rf', $\alpha={angle:g}^\circ$'
                line, = ax.plot(coordinate[coordinate > 0], leading[coordinate > 0],
                                color='C1' if polar == 'numerical' else f'C{index}',
                                linestyle='--' if polar == 'numerical' else '-',
                                linewidth=STYLE['lines.linewidth'],
                                zorder=2, label=label)
                np.testing.assert_array_equal(line.get_ydata(), leading[positive])
            ax.set(xscale='log', yscale='log')
            ax.set_xticks([.05, .1, .2, .3, .5, .7, .9])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.4, which='major', linestyle=':', linewidth=.8)
            ax.legend(fontsize=9, ncol=2, loc='lower left')
        axes[0].set(xlim=(.05, x_upper), ylim=(min(visible_values)/1.2, max(visible_values)*1.2))
        fig.supxlabel(r'$St$ [-]', fontsize=STYLE['axes.labelsize'])
        fig.supylabel(r'$\lambda_1/\lambda_{\mathrm{ref}}$ [-]', fontsize=STYLE['axes.labelsize'])
        leading_stem = destination/'spod_leading_eigenvalues'
        export(fig, leading_stem)
        plt.close(fig)
    leading_report = dict(report)
    leading_report.update(size_mm=[180, 85], strouhal_limits=[.05, x_upper], panel_indices=panel_indices, display='Single panel: PSP 3.5/4/4.5/5/6 and AZDES4; logarithmic St and eigenvalue axes; PSP curves smoothed with raw spectra in background; AZDES unsmoothed',
                          normalization='Divide by maximum leading eigenvalue at St >= 0.2 within displayed range; all PSP curves share PSP alpha4 reference; AZDES uses its own reference',
                          normalization_band_St=[.2, x_upper], normalization_references=peak_references,
                          panels=[dict(item, normalization_factor=factor) for item, factor in zip(metadata, peak_factors)],
                          resolution_markers='none',
                          smoothing={'datasets': 'PSP only; AZDES unchanged', 'raw_overlay': {'linewidth_pt': .55, 'alpha': .3, 'color': 'same as smoothed PSP curve', 'zorder': 1}, 'method': 'Gaussian on uniform log10(St) grid, linear amplitudes, reflected edges, interpolate back to native positive bins', 'sigma_decades': .02, 'kernel_truncation_sigma': 4, 'log_grid_points': 4096, 'normalization': 'unchanged raw reference peaks; no renormalization after smoothing'},
                          curve_validation='Raw reference peaks checked against saved data; plotted curves match smoothing outputs; AZDES matches raw normalized leading eigenvalue exactly')
    leading_report.pop('reference_grid_lines_St')
    leading_stem.with_suffix('.json').write_text(json.dumps(leading_report, indent=2)+'\n')
    print(leading_stem.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
