"""Exact pooled Cp statistics for automatic, Mask 2 and cropped PSP domains."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import resource
import time

from dotenv import load_dotenv
import numpy as np
import torch
from flowtorch.data import PSPDataloader
from convert_surface import ROOT, configured_path, sha256
from psp_study import save_json


def combined_masks(automatic, native, crop=10):
    interior=np.zeros(native.shape,dtype=bool)
    if min(native.shape)<=2*crop:raise ValueError('Crop removes the whole domain')
    interior[crop:-crop,crop:-crop]=True
    return [('automatic',automatic),('automatic_mask2',automatic & native),
            ('automatic_mask2_crop10',automatic & native & interior)]


def pooled(data, mask):
    """Equal-weight pixel-time samples; exact linear-interpolation quantiles."""
    values=np.ascontiguousarray(data[mask]).reshape(-1)
    if values.dtype!=np.float64 or not np.isfinite(values).all():
        raise ValueError('Expected finite Float64 retained values')
    if not values.size:raise ValueError('Empty retained domain')
    result={'samples':int(values.size),'minimum':float(values.min()),
            'maximum':float(values.max()),'mean':float(values.mean(dtype=np.float64))}
    q=np.quantile(values,[.01,.99],method='linear',overwrite_input=True)
    result.update(q01=float(q[0]),q99=float(q[1]))
    assert result['minimum']<=result['q01']<=result['q99']<=result['maximum']
    return result


def smoke():
    data=(np.arange(6*7*11,dtype=np.float64).reshape(6,7,11)**1.3)/17
    automatic=np.ones((6,7),bool);automatic[0,0]=False
    native=np.ones_like(automatic);native[2,2]=False
    for _,mask in combined_masks(automatic,native,crop=1):
        actual=pooled(data,mask);ordered=sorted(data[mask].ravel().tolist());n=len(ordered)
        def reference(q):
            index=(n-1)*q;lo=int(index);fraction=index-lo
            return ordered[lo]*(1-fraction)+ordered[min(lo+1,n-1)]*fraction
        np.testing.assert_allclose([actual['minimum'],actual['maximum'],actual['mean'],actual['q01'],actual['q99']],
            [ordered[0],ordered[-1],sum(ordered)/n,reference(.01),reference(.99)],rtol=1e-13)
    _,intersection,cropped=combined_masks(automatic,native,crop=1)
    assert not intersection[1][2,2] and not cropped[1][0,:].any() and not cropped[1][:,-1].any()
    return {'status':'pass','reference':'Python sorted samples with explicit linear interpolation; unequal values, mask intersection and edge exclusion'}


def calculate(directory,destination,limit=None):
    start=time.perf_counter();prior=json.loads((directory/'run.json').read_text())
    source=configured_path('PSP_DATA_DIR')/prior['source_file']
    fingerprint={'code':sha256(Path(__file__)),'source':sha256(source),
        'masks':sha256(directory/'masks.npz'),'fields':sha256(directory/'fields.npz')}
    if fingerprint['source']!=prior['source_sha256']:raise ValueError('Original dataset has changed')
    if destination.exists():
        cached=json.loads(destination.read_text())
        if cached.get('fingerprints')==fingerprint and cached.get('snapshot_limit')==limit and cached['status']=='complete':
            print(f'{prior["polar"]}: reused compatible global statistics',flush=True);return cached
        raise ValueError(f'Incompatible existing result {destination}')
    loader=PSPDataloader(str(source),dtype=torch.float64)
    try:
        loader.zone=prior['zone'];times=loader.write_times[:limit] if limit else loader.write_times
        data=loader.load_snapshot('Cp',times).numpy()
        assert data.dtype==np.float64 and data.shape[-1]==len(times)
        if limit is None:assert len(times)==prior['snapshots']==3639
        with np.load(directory/'fields.npz') as fields:
            native=fields['native_Mask2'];finite=fields['finite_keep']
        with np.load(directory/'masks.npz') as masks:
            rows=[]
            for label in masks.files:
                if not label.startswith('p') or '_t' not in label:continue
                auto=masks[label];assert auto.dtype==bool and not (auto & ~finite).any()
                previous=None
                for policy,mask in combined_masks(auto,native):
                    count=int(mask.sum())
                    if previous is not None:assert count<=previous
                    previous=count
                    result=pooled(data,mask)
                    assert result['samples']==count*len(times)
                    rows.append({'polar':prior['polar'],'angle_deg':prior['angle_of_attack_deg'],
                        'mask':label,'policy':policy,'pixels':count,'retained_pixel_percent':100*count/mask.size,**result})
            assert len(rows)==27
        record={'status':'complete','polar':prior['polar'],'snapshots':len(times),'snapshot_limit':limit,
            'fingerprints':fingerprint,'definition':'Equal-weight pooled retained pixel-time Cp samples; NumPy linear quantiles; no area weighting',
            'crop_pixels_each_edge':10,'mask2_semantics':'Positive native Mask2 values retained, intersected with automatic mask',
            'dtype':str(data.dtype),'versions':{k:importlib.metadata.version(k) for k in ['numpy','torch','flowtorch-fluid']},
            'seconds':time.perf_counter()-start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'rows':rows}
        if record['seconds']>900:raise TimeoutError('Per-polar 900-second budget exceeded')
        save_json(destination,record)
        print(f'{prior["polar"]}: 27 pooled summaries, {record["seconds"]:.1f}s, peak RSS {record["peak_process_rss_bytes"]/1e9:.2f} GB',flush=True)
        return record
    finally:loader._file.close()


def report(records,destination):
    rows=[row for record in records for row in record['rows']]
    with (destination/'global_mask_statistics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Global PSP mask comparison','',
        'All 3,639 snapshots per polar. Each statistic pools all retained pixel-time Cp values with equal weight. Quantiles use exact linear interpolation, not averages of per-time quantiles. All calculations use Float64. Automatic masks use the maximum reduction over the full record. Mask 2 retains positive supplied mask values; crop10 additionally excludes ten pixels on each of the four image edges.','',
        'Policies: A = automatic mask; B = A intersected with Mask 2; C = B plus crop10. Pixel retention is relative to the original 465 × 159 image.','']
    names={'automatic':'A','automatic_mask2':'B','automatic_mask2_crop10':'C'}
    for record in records:
        lines += [f'## Polar {record["polar"]} ({record["rows"][0]["angle_deg"]:g} degrees)','',
            '| Mask | Policy | Pixels | Retained % | Minimum | 1% | Mean | 99% | Maximum |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in record['rows']:
            lines.append(f'| {r["mask"]} | {names[r["policy"]]} | {r["pixels"]} | {r["retained_pixel_percent"]:.2f} | '+ ' | '.join(f'{r[k]:.6g}' for k in ['minimum','q01','mean','q99','maximum'])+' |')
        lines.append('')
    (destination/'global_mask_statistics.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true');parser.add_argument('--polar')
    args=parser.parse_args();load_dotenv(ROOT/'.env');torch.set_num_threads(4)
    output=configured_path('OUTPUT_DIR')/'psp/working';destination=output/'global_statistics';destination.mkdir(exist_ok=True)
    save_json(destination/'smoke.json',smoke())
    directories=sorted(output.glob('polar_*'))
    if args.polar:directories=[p for p in directories if p.name==f'polar_{args.polar}']
    if args.pilot:directories=directories[:1]
    for directory in directories:
        calculate(directory,destination/f'{directory.name}{"_pilot" if args.pilot else ""}.json',32 if args.pilot else None)
    if not args.pilot:
        records=[json.loads(p.read_text()) for p in sorted(destination.glob('polar_*.json')) if '_pilot' not in p.stem]
        report(records,destination)

if __name__=='__main__':main()
