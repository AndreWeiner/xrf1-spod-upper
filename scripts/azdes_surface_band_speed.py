"""Apply the PSP band phase-speed estimator to the numerical mean shock."""
import argparse
import json
import h5py
import numpy as np
import torch
from dotenv import load_dotenv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from flowtorch.data import map_points_to_grid_2d
from convert_surface import ROOT, configured_path, sha256
from mode_similarity_study import read_case, prefix_basis
from plot_surface_band_gradient import compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-slope',type=float,default=2.)
    parser.add_argument('--spanwise-component',action='store_true')
    args = parser.parse_args()
    load_dotenv(ROOT/'.env')
    torch.set_num_threads(4)
    output = configured_path('OUTPUT_DIR')
    base = output/'psp'
    working = base/'working/displacement_directional'
    numerical = base/'paper/numerical_overview'
    record = json.loads((numerical/'preparation.json').read_text())
    source = numerical/'fields.npz'
    geometry = output/'surface/geometry.h5'
    assert sha256(source)==record['output_sha256']
    assert sha256(geometry)==record['source_geometry_sha256']
    step = record['refinement']
    with np.load(source) as data:
        xyz = data['coordinates'][::step,::step].copy()/1000
        front = data['front'][::step].copy()
    span = (-front[:,1]+xyz[...,1].max())/np.ptp(xyz[...,1])
    keep = np.isfinite(front).all(axis=1)&(span>=.4)
    front,span = front[keep],span[keep]
    order = np.argsort(span)
    front,span = front[order],span[order]
    assert np.all(np.diff(span)>0)

    tri = Delaunay(xyz[...,:2].reshape(-1,2))
    normal = np.cross(np.gradient(xyz,axis=0),np.gradient(xyz,axis=1))
    normals = LinearNDInterpolator(tri,normal.reshape(-1,3))(front[:,:2])
    cross = np.cross(normals,np.gradient(front,axis=0))
    cross /= np.linalg.norm(cross,axis=1)[:,None]
    cross *= np.where(cross[:,0]>=0,1,-1)[:,None]
    fit = json.loads((ROOT/'config/psp_shock_displacement.json').read_text())
    offsets = np.linspace(-.004,.004,fit['normal_samples'])
    targets = (front[:,None,:]+cross[:,None,:]*offsets[None,:,None])[...,:2]
    sampled = LinearNDInterpolator(tri,xyz.reshape(-1,3))(targets)
    assert np.isfinite(sampled).all()
    distances = np.linalg.norm(np.diff(sampled,axis=1),axis=-1)
    weights = np.zeros(sampled.shape[:2])
    weights[:,:-1] += distances/2
    weights[:,1:] += distances/2
    case = read_case(output,'azdes')
    prefix_basis(case,working)
    with h5py.File(geometry) as data:
        points = data['zones/z000000/points'][:,:2]
    coords = [torch.from_numpy(xyz[...,i].copy()) for i in range(3)]
    source_x = points[:,0]-points[:,0].min()
    target_x = xyz[...,0]-xyz[...,0].min()
    hull = Delaunay(np.column_stack((source_x,points[:,1])))
    band_xy = np.stack([targets[...,0]-xyz[...,0].min(),-targets[...,1]],axis=-1)
    inside = hull.find_simplex(band_xy.reshape(-1,2)).reshape(weights.shape)>=0
    assert inside.all(), 'Numerical band leaves the native surface footprint'
    mapped = map_points_to_grid_2d(torch.from_numpy(source_x),torch.from_numpy(points[:,1].copy()),
                                  case['basis'],torch.from_numpy(target_x),-coords[1])
    case['basis'] = mapped.reshape(-1,case['rank'])
    case['indices'] = np.arange(case['basis'].shape[0])
    st = case['St']
    ids = np.flatnonzero((st>=.1)&(st<=.7))
    compute(working,'azdes',st[ids],span,{'target':targets,'coordinates':sampled,'weights':weights},
            case['coeff'][:,ids],case,coords,np.ones(xyz.shape[:2],bool),tri,
            fit['mean_and_mode_smoothing_sigma_pixels'],fit['smoothing_truncate'],
            sign_filter=False,amplitude_filter=False,segments=48,slope_only=True,st_max=.7,minimum_slope=args.minimum_slope,spanwise_component=args.spanwise_component)
    (working/f'azdes_surface_band_speed_metadata_minslope{args.minimum_slope:g}.json').write_text(json.dumps({
        'mode_source':case['source'],'shock_sha256':sha256(source),
        'geometry_sha256':sha256(geometry),'band_inside_native_xy_hull':bool(inside.all()),
        'mapping':'Numerical overview alignment. Native AZDES basis interpolated onto unrefined PSP geometry before differentiation.',
        'measured_span_range':span[[0,-1]].tolist(),'frequency_range':st[ids][[0,-1]].tolist()
    },indent=2)+'\n')


if __name__=='__main__':
    main()
