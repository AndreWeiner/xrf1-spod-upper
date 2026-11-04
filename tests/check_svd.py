"""Independent weighted SVD oracle and saved-factor PAMSPOD reuse checks."""
from pathlib import Path
import sys
import tempfile
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from flowtorch.analysis import SVD, PAMSPOD
from flowtorch.analysis.state_vector import StateVectorSource, StateVectorLayout, FieldSpec

class SyntheticSource(StateVectorSource):
    def __init__(self):
        torch.manual_seed(712)
        self.x=torch.randn(3,31,24,dtype=torch.float64)+torch.arange(31,dtype=torch.float64)[None,:,None]
        self.w=torch.linspace(.1,2,31,dtype=torch.float64)
        self._layout=StateVectorLayout([FieldSpec(f) for f in ['cfx','cfy','cfz']],(31,))
    @property
    def layout(self):return self._layout
    @property
    def n_snapshots(self):return 24
    def read(self,s,t):return self.x[:,s,t].reshape(-1,self.x[:,s,t].shape[-1])
    def read_weight(self,s):return self.w[s]

def main():
    torch.set_num_threads(4)
    source=SyntheticSource()
    with tempfile.TemporaryDirectory() as tmp:
        svd=SVD(source,rank=23,subtract_mean=True,spatial_batch_size=11,basis_directory=tmp)
        x=source.x.reshape(93,24);centered=x-x.mean(1,keepdim=True);w=source.w.repeat(3)
        reference=np.linalg.svd((w.sqrt()[:,None]*centered).numpy(),full_matrices=False)
        np.testing.assert_allclose(svd.s_full.numpy(),reference[1],rtol=1e-10,atol=1e-12)
        u=svd.U.materialize_local()
        np.testing.assert_allclose((u.T@(w[:,None]*u)).numpy(),np.eye(svd.rank),atol=1e-10)
        np.testing.assert_allclose((u@(svd.s[:,None]*svd.V.T)).numpy(),centered.numpy(),atol=1e-10)
        svd.save(Path(tmp)/'checkpoint')
        restored=SVD.load(Path(tmp)/'checkpoint',basis_directory=tmp)
        direct=PAMSPOD(x,.1,rank=23,weight=w,subtract_mean=True,adaptive=False,max_tapers=4)
        reused=PAMSPOD.from_svd(restored,.1,adaptive=False,max_tapers=4)
        np.testing.assert_allclose(reused.eigvals.numpy(),direct.eigvals.numpy(),rtol=1e-9,atol=1e-11)
        svd.close();restored.close()
    print(json.dumps({'status':'passed','checks':['nonuniform weighted centered SVD vs NumPy','weighted orthogonality','exact centered reconstruction','checkpoint restore and from_svd spectral reuse vs direct PAMSPOD']}))

if __name__=='__main__':main()
