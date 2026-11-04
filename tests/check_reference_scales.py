"""Check the central scales against the prior reference workflow and SI conversion."""
from pathlib import Path
import ast,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from reference_scales import reference,strouhal


def main():
    reference_values={'0255':214.459,'0256':214.374,'0257':214.479,'0258':214.422,
                      '0259':214.487,'0260':214.337,'0261':214.364,'0262':214.363}
    for polar,velocity in reference_values.items():
        chord,u=reference('psp',polar)
        assert chord==.1965 and u==velocity
        np.testing.assert_allclose(strouhal([0,100,1000],'psp',polar),np.array([0,100,1000])*.1965/velocity,rtol=1e-15,atol=0)
    chord,u=reference('numerical');assert chord==.1965 and u==214.422
    assert strouhal([1000],'numerical').dtype==np.float64
    np.testing.assert_allclose(strouhal([1000],'numerical'),[0.9164171586866533],rtol=1e-12)
    f=np.array([0,12.3,300,8064.5],dtype=np.float64)
    np.testing.assert_allclose(strouhal(f,'numerical')*u/chord,f,rtol=1e-15,atol=0)
    try:reference('psp','9999')
    except KeyError:pass
    else:raise AssertionError('Missing polar must not silently use a fallback velocity')
    print('Central scales, per-polar conversion, round trip and missing-polar checks passed.')

if __name__=='__main__':main()
