"""Map numerical Cp snapshots to a refined PSP grid and extract paper markers."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as functional
from dotenv import load_dotenv
from flowtorch.data import HDF5Dataloader, map_points_to_grid_2d
from convert_surface import ROOT, configured_path, sha256
from extract_psp_mean_shocks import repair_and_gradient, extract_front
from extract_psp_trailing_edge import extract_section


def refine(coordinates,factor):
    a=torch.as_tensor(coordinates,dtype=torch.float64).permute(2,0,1).unsqueeze(0)
    shape=tuple((n-1)*factor+1 for n in coordinates.shape[:2])
    return functional.interpolate(a,size=shape,mode='bilinear',align_corners=True)[0].permute(1,2,0).numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement',type=int,default=4)
    parser.add_argument('--chunk-size',type=int,default=32)
    parser.add_argument('--limit',type=int,help='Separate bounded smoke run')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if min(args.refinement,args.chunk_size)<1:parser.error('Positive refinement/chunk size required')
    if args.limit is not None and (args.output_dir is None or args.limit<2):parser.error('Smoke limit needs >=2 snapshots and explicit output directory')
    torch.set_num_threads(4);load_dotenv(ROOT/'.env')
    base=configured_path('OUTPUT_DIR');surface=base/'surface'
    destination=args.output_dir or base/'psp/paper/numerical_overview'
    destination.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    psp=base/'psp/working/polar_0258/fields.npz'
    with np.load(psp) as f:coordinates=refine(f['coordinates']/1000,args.refinement)
    # Identical alignment to legacy azdes_data_utils.normalized_*_coordinates:
    # numerical positive y matches PSP -y; both x origins shifted to their minimum.
    tx=torch.from_numpy(coordinates[...,0]-coordinates[...,0].min())
    ty=torch.from_numpy(-coordinates[...,1])
    conversion=json.loads((surface/'conversion.json').read_text())
    if conversion['status']!='complete':raise ValueError('Incomplete numerical input')
    with HDF5Dataloader(str(surface),dtype=torch.float64,zones=['wing-upper']) as loader:
        points=loader.vertices.numpy();sx=torch.from_numpy(points[:,0]-points[:,0].min());sy=torch.from_numpy(points[:,1])
        times=loader.write_times[:args.limit] if args.limit else loader.write_times
        mean=np.zeros(tx.shape);m2=np.zeros(tx.shape);count=0
        for start in range(0,len(times),args.chunk_size):
            selected=times[start:start+args.chunk_size]
            cp=loader.load_snapshot('cp',selected)
            mapped=map_points_to_grid_2d(sx,sy,cp,tx,ty).numpy()
            if mapped.ndim==2:mapped=mapped[...,None]
            if not np.isfinite(mapped).all():raise ValueError('Nonfinite mapped Cp')
            # Stable batch-combined population moments; no snapshot cube retained.
            n=mapped.shape[-1];batch_mean=mapped.mean(axis=-1)
            batch_m2=((mapped-batch_mean[...,None])**2).sum(axis=-1)
            delta=batch_mean-mean;total=count+n
            m2+=batch_m2+delta**2*(count*n/total)
            mean+=delta*(n/total);count=total
            if start==0 or total%256==0 or total==len(times):
                print(f'Mapped {total}/{len(times)} snapshots on {tuple(tx.shape)}; {time.perf_counter()-started:.1f} s',flush=True)
            del mapped,cp
    keep=np.ones(mean.shape,dtype=bool)
    _,gradient=repair_and_gradient(mean,coordinates,keep,sigma=float(args.refinement),outlier_threshold=None)
    front=extract_front(gradient,coordinates,keep,crop=5*args.refinement,spline_rms_m=.001)
    section=extract_section(mean,coordinates,keep,chord_fraction=.95,search=(.2,.65),sigma=2.*args.refinement)
    np.savez_compressed(destination/'fields.npz',coordinates=coordinates*1000,mean=mean,variance=m2/count,
        front=front['front_xyz_m'],raw_front=front['raw_xyz_m'],independent_front=front['independent_peak_xyz_m'],
        dcp_dx_per_m=gradient,marker=section['marker_xyz_m'],section_xyz_m=section['section_xyz_m'],
        section_cp=section['mean_cp'],section_filtered_cp=section['filtered_cp'],
        section_dcp_dspan_per_m=section['dcp_dspan_per_m'])
    record=dict(status='complete',snapshots=count,angle_deg=4.,grid_shape=list(mean.shape),refinement=args.refinement,
        seconds=time.perf_counter()-started,mean_range=[float(mean.min()),float(mean.max())],
        variance_range=[float((m2/count).min()),float((m2/count).max())],
        separation_span_fraction=float(section['span_fraction'][section['index']]),
        marker_xyz_m=section['marker_xyz_m'].tolist(),
        alignment='Legacy: x_num-min(x_num) -> x_PSP-min(x_PSP); y_num -> -y_PSP. Physical target coordinates restored for plotting.',
        moments='Population temporal variance of interpolated snapshots, stable batch-combined moments, float64',
        shock='Same continuous positive planar chordwise-gradient ridge; Gaussian sigma refinement grid samples; no derivative outlier replacement for numerical field; 1-mm RMS spline; border exclusion 5 native cells',
        separation='Same 95%-chord section, absolute spanwise derivative maximum in 20-65% of displayed span; Gaussian sigma 2 native span intervals',
        mapping='FlowTorch linear scattered-point interpolation with nearest-neighbor fallback beyond source convex hull; bilinear PSP refinement preserving original nodes; no PSP mask applied',
        source_conversion_sha256=sha256(surface/'conversion.json'),source_geometry_sha256=sha256(surface/'geometry.h5'),
        source_fields=conversion['outputs'].get('fields.h5'),psp_fields_sha256=sha256(psp),
        script_sha256=sha256(Path(__file__)),output_sha256=sha256(destination/'fields.npz'))
    (destination/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(destination,flush=True)


if __name__=='__main__':main()
