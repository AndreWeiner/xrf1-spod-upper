"""Extract mean PSP compression fronts using FlowTorch repair and gradients.

Native coordinates are mm; derivatives and saved fronts use metres. The planar
x derivative is taken at fixed y, rather than a grid-index derivative. Native
rows approximate span stations but have slight chordwise y variation.
"""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from scipy.interpolate import UnivariateSpline
from flowtorch.data import replace_masked_values, curvilinear_gradient
from convert_surface import ROOT, configured_path, sha256

POLARS = ['0257', '0258', '0259', '0260', '0261']


def repair_and_gradient(mean, coordinates, keep, sigma=1., outlier_threshold=3.5):
    """Keep valid measurements exactly; interpolate exclusions before filtering."""
    x, y = (torch.as_tensor(coordinates[..., k], dtype=torch.float64) for k in (0, 1))
    values = torch.as_tensor(mean, dtype=torch.float64)
    valid = torch.as_tensor(keep & np.isfinite(mean))
    repaired = replace_masked_values(x, y, values, valid)
    if not torch.isfinite(repaired).all():
        raise ValueError('Repair did not produce a finite mean')
    np.testing.assert_array_equal(repaired.numpy()[valid], mean[valid])
    gradient = curvilinear_gradient(repaired, x, y, smoothing_sigma=sigma or None,
                                   outlier_threshold=outlier_threshold,
                                   outlier_window_size=3, edge_order=2)
    return repaired.numpy(), gradient[..., 0].numpy()


def extract_front(gradient, coordinates, keep, search=(.10, .85), crop=5, spline_rms_m=0.):
    """Continuous positive-gradient ridge; missing rows remain NaN.

    Search coordinates use the original row's leading/trailing-edge x limits.
    Spline smoothing is optional and never bridges missing-row gaps.
    """
    x = coordinates[..., 0]
    xi = (x-x[:, :1])/(x[:, -1:]-x[:, :1])
    if not np.all(np.diff(x, axis=1) > 0):
        raise ValueError('Expected increasing chordwise x along grid axis 1')
    candidate = np.isfinite(gradient) & (gradient > 0) & (xi >= search[0]) & (xi <= search[1])
    if crop:
        candidate[:crop] = False; candidate[-crop:] = False
        candidate[:, :crop] = False; candidate[:, -crop:] = False
    found = candidate.any(axis=1)
    indices = np.argmax(np.where(candidate, gradient, -np.inf), axis=1)
    independent_indices = indices.copy()
    # Same normalized-gradient / bounded-step objective as the legacy
    # skander_mean_bl_utils.continuous_shock_ridge; no field smoothing here.
    for segment in np.split(np.flatnonzero(found), np.flatnonzero(np.diff(np.flatnonzero(found)) > 1)+1):
        if not segment.size:
            continue
        score = np.where(candidate[segment], gradient[segment], 0.)
        score /= np.maximum(score.max(axis=1, keepdims=True), np.finfo(float).eps)
        positions = np.arange(x.shape[1])
        step = np.abs(positions[:, None]-positions[None, :])
        transition = np.where(step <= 4, .018*step, np.inf)
        cost = np.where(candidate[segment[0]], -score[0], np.inf)
        parents = np.zeros(score.shape, dtype=int)
        for i in range(1, len(segment)):
            costs = cost[:, None]+transition
            parents[i] = costs.argmin(axis=0)
            cost = costs[parents[i], positions]-score[i]
            cost[~candidate[segment[i]]] = np.inf
        if not np.isfinite(cost).any():
            raise ValueError('No continuous shock path through candidate rows')
        indices[segment[-1]] = cost.argmin()
        for i in range(len(segment)-1, 0, -1):
            indices[segment[i-1]] = parents[i, indices[segment[i]]]
    rows = np.arange(x.shape[0])
    independent_points = coordinates[rows, independent_indices].copy()
    independent_points[~found] = np.nan
    points = coordinates[rows, indices].copy()
    points[~found] = np.nan
    peak = np.where(found, gradient[rows, indices], np.nan)
    selected_xi = np.where(found, xi[rows, indices], np.nan)
    repaired_peak = found & ~keep[rows, indices]
    smooth = points.copy()
    # Fit x as a function of physical span; retain raw peak points separately.
    for segment in np.split(np.flatnonzero(found), np.flatnonzero(np.diff(np.flatnonzero(found)) > 1)+1):
        if spline_rms_m <= 0 or segment.size < 4:
            continue
        span = -points[segment, 1]
        if not np.all(np.diff(span) > 0):
            raise ValueError('Nonmonotone span coordinates for spline')
        fit = UnivariateSpline(span, points[segment, 0], s=segment.size*spline_rms_m**2)
        smooth[segment, 0] = fit(span)
        # Re-interpolate y,z on each native section to keep points on the grid.
        for row in segment:
            smooth[row, 1:] = [np.interp(smooth[row, 0], x[row], coordinates[row, :, k]) for k in (1, 2)]
    if np.any(found & ((smooth[:, 0] < x[:, 0]) | (smooth[:, 0] > x[:, -1]))):
        raise ValueError('Spline left the wing; reduce smoothing')
    return dict(independent_peak_xyz_m=independent_points, raw_xyz_m=points, front_xyz_m=smooth, found=found,
                chord_index=np.where(found, indices, -1), chord_fraction=selected_xi,
                peak_dcp_dx_per_m=peak, peak_was_repaired=repaired_peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sigma', type=float, default=1., help='Gaussian sigma in grid-index units')
    parser.add_argument('--outlier-threshold', type=float, default=3.5)
    parser.add_argument('--search', nargs=2, type=float, default=[.10, .85], metavar=('LOW', 'HIGH'))
    parser.add_argument('--spline-rms-mm', type=float, default=0., help='Optional RMS spline smoothing budget; zero disables')
    args = parser.parse_args()
    if args.sigma < 0 or args.spline_rms_mm < 0 or not 0 < args.search[0] < args.search[1] < 1:
        parser.error('Nonnegative smoothing and 0 < search low < high < 1 required')
    torch.set_num_threads(4)
    load_dotenv(ROOT/'.env')
    source = configured_path('OUTPUT_DIR')/'psp/working'
    destination = args.output_dir or configured_path('OUTPUT_DIR')/'psp/paper/mean_shocks'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for polar in POLARS:
        directory = source/f'polar_{polar}'
        run = json.loads((directory/'run.json').read_text())
        if run['status'] != 'complete': raise ValueError(f'Incomplete polar {polar}')
        with np.load(directory/'fields.npz') as fields, np.load(directory/'masks.npz') as masks:
            coordinates = fields['coordinates'].copy()/1000
            mean = fields['mean'].copy()
            keep = masks['p5_t10'].copy() & np.isfinite(mean)
            keep[:5] = False; keep[-5:] = False; keep[:, :5] = False; keep[:, -5:] = False
        repaired, gradient = repair_and_gradient(mean, coordinates, keep, args.sigma, args.outlier_threshold)
        # Finite-only reference makes the effect of automatic masking inspectable.
        reference_mean, reference_gradient = repair_and_gradient(mean, coordinates, np.isfinite(mean), args.sigma, args.outlier_threshold)
        front = extract_front(gradient, coordinates, keep, args.search, spline_rms_m=args.spline_rms_mm/1000)
        reference = extract_front(reference_gradient, coordinates, np.isfinite(mean), args.search)
        stem = destination/f'polar_{polar}'
        np.savez_compressed(stem.with_suffix('.npz'), coordinates_m=coordinates,
            mean_original=mean, mean_masked=np.where(keep, mean, np.nan), mean_repaired=repaired,
            mean_finite_reference=reference_mean, analysis_keep=keep, dcp_dx_per_m=gradient,
            finite_reference_dcp_dx_per_m=reference_gradient,
            finite_reference_front_xyz_m=reference['raw_xyz_m'], **front)
        with stem.with_suffix('.csv').open('w') as stream:
            writer=csv.writer(stream)
            writer.writerow(['row','found','chord_index','chord_fraction','raw_x_m','raw_y_m','raw_z_m','front_x_m','front_y_m','front_z_m','peak_dcp_dx_per_m','peak_was_repaired'])
            for row in range(mean.shape[0]):
                writer.writerow([row, int(front['found'][row]), front['chord_index'][row], front['chord_fraction'][row],
                    *front['raw_xyz_m'][row], *front['front_xyz_m'][row], front['peak_dcp_dx_per_m'][row], int(front['peak_was_repaired'][row])])
        records.append(dict(polar=polar, angle_deg=run['angle_of_attack_deg'], found=int(front['found'].sum()),
            interpolated_peaks=int(front['peak_was_repaired'].sum()),
            input_sha256={n:sha256(directory/n) for n in ['run.json','fields.npz','masks.npz']},
            npz_sha256=sha256(stem.with_suffix('.npz'))))
        print(polar, 'valid rows:', records[-1]['found'], 'peaks in repaired areas:', records[-1]['interpolated_peaks'], flush=True)
    manifest=dict(settings=vars(args) | {'output_dir':str(destination)},
        definition='Positive planar dcp/dx at fixed y after repair, Gaussian filtering, and local derivative outlier replacement. Continuous ridge maximizes row-normalized positive gradient with index-step penalty 0.018 and max step 4; independent maxima are also saved. Candidate compression front, not proof of a shock at every span station.',
        mask='Full-record p5_t10 plus five-pixel border crop; no native Mask 2',
        coordinate_units='m', derivative_units='1/m', spline='x(-y), separate contiguous valid segments; no extrapolation across gaps',
        source_examples=['mk2a_analysis_skander/analyze_psp_derivatives.py:compute_derivatives', 'mk2a_analysis_skander/analyze_line_samples.py:locate_mean_shocks', 'mk2a_analysis_skander/skander_mean_bl_utils.py:continuous_shock_ridge'],
        flowtorch_version=importlib.metadata.version('flowtorch-fluid'),script_sha256=sha256(Path(__file__)), cases=records)
    (destination/'extraction.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(destination)


if __name__ == '__main__': main()
