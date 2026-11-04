"""Independent doubly centered weighted SVD and saved-factor SPOD equivalence."""
from pathlib import Path
import sys
import tempfile
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from flowtorch.analysis import SVD,PAMSPOD
from psp_svd import CenteredPSPSource,spatial_center,validate


def main():
    torch.set_num_threads(4)
    rng=np.random.default_rng(852)
    x=rng.normal(size=(53,31))+np.linspace(-4,3,53)[:,None]+np.cos(np.arange(31))[None,:]*3
    w=np.geomspace(.1,4,53)
    spatial=np.average(x,axis=0,weights=w)
    centered=x-spatial[None,:];temporal=centered.mean(axis=1)
    reference=centered-temporal[:,None]
    data=torch.from_numpy(x.copy());weight=torch.from_numpy(w)
    actual,residual=spatial_center(data,weight)
    np.testing.assert_allclose(actual.numpy(),spatial,atol=1e-14)
    np.testing.assert_allclose(data.numpy(),centered,atol=1e-14)
    np.testing.assert_allclose(np.average(reference,axis=0,weights=w),0,atol=1e-14)
    # Unequal weights distinguish the chosen spatial mean from an arithmetic one.
    assert np.max(np.abs(spatial-x.mean(0)))>.1
    source=CenteredPSPSource(data,weight)
    with tempfile.TemporaryDirectory() as tmp:
        svd=SVD(source,rank=30,mode='tsqr',subtract_mean=True,spatial_batch_size=17,basis_directory=tmp)
        sigma=np.linalg.svd(np.sqrt(w)[:,None]*reference,compute_uv=False)
        np.testing.assert_allclose(svd.s_full.numpy(),sigma,rtol=1e-10,atol=1e-12)
        evidence=validate(svd,source)
        modes=svd.U.materialize_local().numpy()
        reconstructed=modes @ (svd.s[:,None]*svd.V.T).numpy()
        np.testing.assert_allclose(reconstructed+temporal[:,None]+spatial[None,:],x,atol=1e-12)
        checkpoint=Path(tmp)/'checkpoint';svd.save(checkpoint)
        restored=SVD.load(checkpoint,basis_directory=tmp)
        direct=PAMSPOD(torch.from_numpy(centered),.0005,rank=30,weight=weight,
            subtract_mean=True,adaptive=False,max_tapers=4)
        reused=PAMSPOD.from_svd(restored,.0005,adaptive=False,max_tapers=4)
        np.testing.assert_allclose(reused.eigvals.numpy(),direct.eigvals.numpy(),rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(restored.s.numpy(),svd.s.numpy(),rtol=0,atol=0)
        svd.close();restored.close()
    print(json.dumps({'status':'pass','fixture':'nonuniform weights, nonuniform pointwise temporal means and spatially uniform oscillation',
        'checks':['spatial means vs independent NumPy weighted average','TSQR singular values vs NumPy direct SVD','doubly centered weighted invariants','original data reconstruction after restoring both means','checkpoint reload and PAMSPOD.from_svd equivalence'],
        'validation':evidence}))

if __name__=='__main__':main()
