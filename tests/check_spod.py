"""Check rank-limited reuse against an independently computed truncated SVD."""
from pathlib import Path
import sys,tempfile,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parent)]
from check_svd import SyntheticSource
from surface_spod import RankView
from flowtorch.analysis import SVD,PAMSPOD
import torch,numpy as np

def main():
 torch.set_default_dtype(torch.float64);torch.set_num_threads(4);s=SyntheticSource()
 with tempfile.TemporaryDirectory() as tmp:
  full=SVD(s,rank=23,subtract_mean=True,spatial_batch_size=11,basis_directory=tmp)
  view=RankView(full,12)
  assert view.s.shape==(12,) and view.V.shape==(24,12) and full.s.shape==(23,)
  direct=SVD(s,rank=12,subtract_mean=True,spatial_batch_size=11,basis_directory=tmp)
  a=PAMSPOD.from_svd(view,.1,adaptive=True,max_tapers=12,tolerance=1e-4)
  b=PAMSPOD.from_svd(direct,.1,adaptive=True,max_tapers=12,tolerance=1e-4)
  np.testing.assert_allclose(a.eigvals.numpy(),b.eigvals.numpy(),rtol=1e-9,atol=1e-11)
  gram=torch.zeros((12,12),dtype=torch.float64)
  for (part,u),(_,v) in zip(view.U.iter_chunks(),direct.U.iter_chunks()):
   gram+=u.T@(s.w[part].repeat(3)[:,None]*v)
  np.testing.assert_allclose(abs(gram).numpy(),np.eye(12),atol=1e-9)
  full.close();direct.close()
 print(json.dumps({'status':'passed','checks':['rank view shapes','adaptive spectra equal direct truncated-SVD result','global weighted subspace identity']}))

if __name__=='__main__':main()
