import xarray as xr
import numpy as np

def get_principle_components_and_EOFs(da, nmode=5, eofscaling=2, pcscaling=1):
    """Returns Principle Components (PC) and EOFs as xr.DataArray and print how much total variance is explained by nmode modes from an input xr.DataArray."""
    # need to use main eofs branch
    # pip install git+https://github.com/ajdawson/eofs.git
    # todo: xeofs from Niclas
    from eofs.xarray import Eof
    assert "lon" in da.coords, "expects dimension lon and lat"
    assert "lat" in da.coords, "expects dimension lon and lat"
    
    lon2d,lat2d = np.meshgrid(da.lon, da.lat)
    wgts = np.cos(lat2d/180*np.pi)**0.5
    solver = Eof(da, weights=wgts)

    EOF = solver.eofs(neofs=nmode, eofscaling=eofscaling)
    VarEx = solver.varianceFraction(neigs=nmode) * 100
    print(f'the first {nmode} modes explain {round(sum(VarEx.values),2)}% of the total variance of {da.name}')
    
    PC = solver.pcs(npcs=nmode, pcscaling=pcscaling)
    return PC, EOF
