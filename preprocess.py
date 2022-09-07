import xarray as xr
import numpy as np

def get_principle_components_and_EOFs(da, nmode=5, eofscaling=2, pcscaling=1, xeofs=False, xeofs_eof_kwargs={}, xeofs_pcs_kwargs={}):
    """Returns Principle Components (PC) and EOFs as xr.DataArray and print how much total variance is explained by nmode modes from an input xr.DataArray.
    
    Set xeofs=True to use Niclas xeofs package and set xeofs_*_kwargs optionally."""
    assert "lon" in da.coords, "expects dimension lon and lat"
    assert "lat" in da.coords, "expects dimension lon and lat"
    
    lon2d,lat2d = np.meshgrid(da.lon, da.lat)
    wgts = np.cos(lat2d/180*np.pi)**0.5
    
    if xeofs: # from Niclas
        from xeofs.xarray import EOF as xEof
        model = xEof(da, n_modes=nmode, norm=True, dim="time", **xeofs_eof_kwargs)
        model.solve()
        VarEx = model.explained_variance_ratio()

        EOF = model.eofs(eofscaling)
        EOF = EOF.assign_coords(explained_variance=VarEx)
        EOF.coords["explained_variance"].attrs["units"] = "%"
        
        PC = model.pcs(eofscaling, **xeofs_pcs_kwargs)
    else:
        # need to use main eofs branch
        # pip install git+https://github.com/ajdawson/eofs.git
        from eofs.xarray import Eof
        solver = Eof(da, weights=wgts)
        VarEx = solver.varianceFraction(neigs=nmode)
        
        EOF = solver.eofs(neofs=nmode, eofscaling=eofscaling)
        EOF = EOF.assign_coords(explained_variance=VarEx)
        EOF.coords["explained_variance"].attrs["units"] = "%"
        
        PC = solver.pcs(npcs=nmode, pcscaling=pcscaling)
        
        # mode starts with 1 as in eofs
        PC = PC.assign_coords(mode=PC.mode+1)
        EOF = EOF.assign_coords(mode=EOF.mode+1)
        
    print(f'the first {nmode} modes explain {round(sum(VarEx.values * 100),2)}% of the total variance of {da.name}')
    return PC, EOF
