"""Independent moment/quantile/FFT checks and full-record max-mask equivalence."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from scipy.stats import skew,kurtosis
from flowtorch.analysis import snapshot_statistics
from flowtorch.data import create_image_mask
from psp_study import masks_from_scores,spectrum


def main():
 torch.set_num_threads(4)
 torch.manual_seed(12)
 data=torch.randn(9,11,12,dtype=torch.float64)*.2
 data+=torch.linspace(-1,1,9,dtype=torch.float64)[:,None,None]
 data[4,5,10]=40;data[0,0,3]=float('nan')
 mask,diag=create_image_mask(data,patch_size=5,threshold=10,temporal_reduction='max',chunk_size=3)
 score=torch.zeros((9,11),dtype=torch.float64);extreme=torch.zeros((9,11),dtype=torch.bool)
 for start in range(0,12,4):
  _,part=create_image_mask(data[...,start:start+4],patch_size=5,threshold=10,temporal_reduction='max',chunk_size=2)
  score=torch.maximum(score,part.spatial_score);extreme|=part.extreme_mask
 assert torch.equal(mask,(~extreme)&(score<=10))
 assert not mask[4,5] and not mask[0,0]
 weight=torch.linspace(.1,2,99,dtype=torch.float64).reshape(9,11)
 result=snapshot_statistics(data,time=np.arange(12)*.1,spatial_mask=mask,spatial_weight=weight,quantiles=[.1,.5,.9],compute=['moments','spatial'],batch_size=4,accumulator_dtype=torch.float64)
 values=data[mask].numpy();w=weight[mask].numpy()
 for name,expected in [('mean',values.mean(1)),('variance',values.var(1)),('skewness',skew(values,axis=1,bias=True)),('kurtosis',kurtosis(values,axis=1,fisher=False,bias=True))]:
  np.testing.assert_allclose(getattr(result.moments,name)[mask].numpy(),expected,rtol=1e-10,atol=1e-11)
 np.testing.assert_allclose(result.spatial_statistics.mean.numpy(),np.average(values,weights=w,axis=0),rtol=1e-12,atol=1e-12)
 for t in range(12):
  order=np.argsort(values[:,t]);v=values[order,t];weights=w[order];positions=(np.cumsum(weights)-.5*weights)/weights.sum()
  np.testing.assert_allclose(result.spatial_statistics.quantiles[:,t].numpy(),np.interp([.1,.5,.9],positions,v),rtol=1e-11,atol=1e-12)
 fs=1000;n=1000;t=np.arange(n)/fs
 fft=spectrum(3+2*np.sin(2*np.pi*50*t),fs)
 peak=np.argmax(fft['hann_amplitude'][1:])+1
 assert fft['frequency_hz'][peak]==50
 np.testing.assert_allclose(fft['hann_amplitude'][peak],2,rtol=1e-5)
 print(json.dumps({'status':'passed','checks':['max reduction and all-snapshot anomaly rejection','nonuniform area-weighted mean and quantiles vs NumPy','temporal moments vs NumPy/SciPy','Hann amplitude and frequency of known sinusoid']}))

if __name__=='__main__':main()
