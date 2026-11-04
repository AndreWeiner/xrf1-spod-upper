"""Verify saved PSP SVD checkpoints, preprocessing and complete file fingerprints."""
import json
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
from convert_surface import ROOT,configured_path,sha256
from psp_study import save_json


def main():
    load_dotenv(ROOT/'.env');working=configured_path('OUTPUT_DIR')/'psp/working';records={}
    directories=sorted((working/'svd').glob('polar_*'))
    if len(directories)!=8:raise ValueError('All eight checkpoints are required')
    for directory in directories:
        record=json.loads((directory/'run.json').read_text());assert record['status']=='complete'
        checkpoint=directory/'checkpoint'
        assert sha256(checkpoint/'metadata.pt')==record['checkpoint_metadata_sha256']
        metadata=torch.load(checkpoint/'metadata.pt',weights_only=True)
        assert metadata['rank']==3638 and metadata['cols']==3639 and metadata['subtract_mean']
        for key in ['s','s_full','V']:
            assert metadata[key].dtype==torch.float64 and bool(torch.isfinite(metadata[key]).all())
        with np.load(directory/'spectrum.npz') as spectrum:
            np.testing.assert_array_equal(metadata['s_full'].numpy(),spectrum['singular_values'])
        with np.load(directory/'preprocessing.npz') as arrays:
            weights=arrays['area_weights'];temporal=arrays['temporal_mean_after_spatial_subtraction'];mask=arrays['mask']
            assert weights.dtype==np.float64 and arrays['snapshot_spatial_mean'].dtype==np.float64
            assert arrays['snapshot_spatial_mean'].shape==(3639,)
            assert not mask[:5].any() and not mask[-5:].any() and not mask[:,:5].any() and not mask[:,-5:].any()
            np.testing.assert_array_equal(arrays['retained_flat_indices'],np.flatnonzero(mask))
        with np.load(working/directory.name/'fields.npz') as fields:
            np.testing.assert_array_equal(weights,fields['area_weights'][mask])
        with np.load(working/directory.name/'masks.npz') as masks:
            expected=masks['p5_t10'].copy();expected[:5]=False;expected[-5:]=False;expected[:,:5]=False;expected[:,-5:]=False
            np.testing.assert_array_equal(mask,expected)
        index=torch.load(checkpoint/'rank-0/index.pt',weights_only=True)
        rows=0
        for left,right,names in index['chunks']:
            assert left==rows
            for key,name in names.items():
                if name is None:continue
                value=np.load(checkpoint/'rank-0'/name,mmap_mode='r')
                assert value.dtype==np.float64 and np.isfinite(value).all()
                if key=='U':assert value.shape==(right-left,3638)
                if key=='weight':np.testing.assert_array_equal(value,weights[left:right])
                if key=='mean':np.testing.assert_allclose(value,temporal[left:right],rtol=1e-12,atol=1e-12)
            rows=right
        assert rows==record['points']==int(mask.sum())
        manifest={str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='verification.json'}
        evidence={'status':'pass','polar':record['polar'],'points':rows,
            'checks':['Default automatic mask plus crop5 only; no Mask 2','Saved weights exactly equal selected existing FlowTorch nodal areas','Native Float64 factors finite with expected dimensions','Stored temporal mean matches preprocessing record','All physical-basis shards present and contiguous','Saved full spectrum equals native checkpoint spectrum'],
            'manifest_sha256':manifest}
        save_json(directory/'verification.json',evidence);records[record['polar']]={'status':'pass','checkpoint_bytes':sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),'validation':record['validation']}
        print(record['polar'],'checkpoint passed',flush=True)
    save_json(working/'svd/validation.json',{'status':'pass','polars':records,'checkpoint_bytes_total':sum(r['checkpoint_bytes'] for r in records.values())})

if __name__=='__main__':main()
