"""Compute the PSP along-shock phase velocities used in the comparative figures."""
import argparse
import json

import numpy as np
import torch
from dotenv import load_dotenv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from convert_surface import ROOT, configured_path, sha256
from mode_similarity_study import prefix_basis, read_case
from plot_surface_band_gradient import compute

POLARS = ('0257', '0258', '0259', '0260', '0261')


def shock_band(xyz, front, found, settings):
    di = np.gradient(xyz, axis=0, edge_order=2)
    dj = np.gradient(xyz, axis=1, edge_order=2)
    normals = np.cross(di, dj)
    normals /= np.linalg.norm(normals, axis=-1)[..., None]

    span = (-front[:, 1] + xyz[..., 1].max()) / np.ptp(xyz[..., 1])
    indices = np.arange(len(front))
    rows = np.flatnonzero(found & np.isfinite(front).all(axis=1) &
                          (span >= settings['span_min']) & (indices >= 5) &
                          (indices < len(front)-5))
    if np.mean(np.diff(-front[rows, 1])) < 0:
        rows = rows[::-1]
    front, span = front[rows], span[rows]
    assert np.all(np.diff(span) > 0)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(front, axis=0), axis=1))]
    tangent = np.gradient(front, arc, axis=0, edge_order=2)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]

    triangulation = Delaunay(xyz[..., :2].reshape(-1, 2))
    normal = LinearNDInterpolator(triangulation, normals.reshape(-1, 3))(front[:, :2])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    tangent -= np.sum(tangent*normal, axis=1)[:, None]*normal
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    cross = np.cross(normal, tangent)
    cross /= np.linalg.norm(cross, axis=1)[:, None]
    cross *= np.where(cross[:, 0] >= 0, 1, -1)[:, None]

    width = settings['selected_half_width_m']
    offset = np.linspace(-width, width, settings['normal_samples'])
    target = (front[:, None, :] + cross[:, None, :]*offset[None, :, None])[..., :2]
    points = LinearNDInterpolator(triangulation, xyz.reshape(-1, 3))(target)
    assert np.isfinite(points).all()
    distance = np.linalg.norm(np.diff(points, axis=1), axis=-1)
    weights = np.zeros(points.shape[:2])
    weights[:, :-1] += distance/2
    weights[:, 1:] += distance/2
    return span, {'target': target, 'coordinates': points, 'weights': weights}, triangulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--polar', choices=POLARS)
    args = parser.parse_args()
    load_dotenv(ROOT/'.env')
    torch.set_num_threads(4)
    output = configured_path('OUTPUT_DIR')
    shocks = output/'psp/paper/mean_shocks'
    records = json.loads((shocks/'extraction.json').read_text())['cases']
    settings = json.loads((ROOT/'config/psp_shock_displacement.json').read_text())
    working = output/'psp/working/displacement_directional'
    working.mkdir(parents=True, exist_ok=True)

    for polar in (args.polar,) if args.polar else POLARS:
        source = shocks/f'polar_{polar}.npz'
        record = next(record for record in records if record['polar'] == polar)
        assert sha256(source) == record['npz_sha256']
        with np.load(source) as data:
            xyz = data['coordinates_m']
            span, band, triangulation = shock_band(xyz, data['front_xyz_m'], data['found'], settings)
        case = read_case(output, polar)
        prefix_basis(case, working)
        np.testing.assert_allclose(xyz.reshape(-1, 3)[case['indices']],
                                   case['coordinates']/1000, rtol=0, atol=1e-14)
        mask = np.zeros(xyz.shape[:2], dtype=bool)
        mask.flat[case['indices']] = True
        coordinates = tuple(torch.from_numpy(xyz[..., index].copy()) for index in range(3))
        st = case['St']
        keep = (st >= .1) & (st <= .7)
        compute(working, polar, st[keep], span, band, case['coeff'][:, keep], case,
                coordinates, mask, triangulation,
                settings['mean_and_mode_smoothing_sigma_pixels'], settings['smoothing_truncate'],
                sign_filter=False, amplitude_filter=False, segments=48, slope_only=True,
                st_max=.7, minimum_slope=4.)


if __name__ == '__main__':
    main()
