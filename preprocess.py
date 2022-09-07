import xarray as xr
import numpy as np

def get_principle_components_and_EOFs(var_anom_sel, nmode=5, eofscaling=2, pcscaling=1):
    """Returns Principle Components (PC) xr.DataArray and print how much total variance is explained by nmode modes."""
    # need to use main branch
    # b
    # todo: xeofs from Niclas
    from eofs.xarray import Eof
    lon2d,lat2d = np.meshgrid(var_anom_sel.lon,var_anom_sel.lat)
    wgts = np.cos(lat2d/180*np.pi)**0.5
    solver = Eof(var_anom_sel, weights=wgts)

    EOF = solver.eofs(neofs=nmode, eofscaling=eofscaling)
    # EOF.plot(col="mode")
    # eigenv = solver.eigenvalues(neigs=nmode)
    VarEx = solver.varianceFraction(neigs=nmode) * 100
    print(f'the first {nmode} modes explain {round(sum(VarEx.values),2)}% of the total variance of {var_anom_sel.name}')
    
    PC = solver.pcs(npcs=nmode, pcscaling=pcscaling)
    return PC, EOF
