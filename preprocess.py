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
        model = xEof(var_anom_sel, n_modes=nmode, weights=wgts, norm=False, dim="time", **xeofs_eof_kwargs)
        model.solve()
        VarEx = model.explained_variance_ratio()
        # scaling ([0, 1, 2]) – EOFs are scaled
        # (i) to be orthonormal (scaling=0)
        # (ii) by the square root of the eigenvalues (scaling=1)
        # (iii) by the singular values (scaling=2).
        # In case no weights were applied, scaling by the singular values results in the EOFs having the unit of the input data (the default is 0).
        EOF = model.eofs(1)
        EOF = EOF.assign_coords(explained_variance=VarEx)
        EOF.coords["explained_variance"].attrs["units"] = "%"
        
        # scaling ([0, 1, 2]) – PCs are scaled
        # (i) to be orthonormal (scaling=0)
        # (ii) by the square root of the eigenvalues (scaling=1)
        # (iii) by the singular values (scaling=2)
        # In case no weights were applied, scaling by the singular values results in the PCs having the unit of the input data (the default is 0).
        PC = model.pcs(0, **xeofs_pcs_kwargs)
    else:
        # need to use main eofs branch
        # pip install git+https://github.com/ajdawson/eofs.git
        from eofs.xarray import Eof
        solver = Eof(da, weights=wgts)
        VarEx = solver.varianceFraction(neigs=nmode)
        # eofscaling: Sets the scaling of the EOFs. The following values are accepted:
        # 0 : Un-scaled EOFs (default).
        # 1 : EOFs are divided by the square-root of their eigenvalues.
        # 2 : EOFs are multiplied by the square-root of their eigenvalues.
        EOF = solver.eofs(neofs=nmode, eofscaling=2)
        EOF = EOF.assign_coords(explained_variance=VarEx)
        EOF.coords["explained_variance"].attrs["units"] = "%"
        
        # pcscaling: Set the scaling of the retrieved PCs. The following values are accepted:
        # 0 : Un-scaled principal components (default).
        # 1 : Principal components are scaled to unit variance (divided by the square-root of their eigenvalue).
        # 2 : Principal components are multiplied by the square-root of their eigenvalue.
        PC = solver.pcs(npcs=nmode, pcscaling=1)
        
        # mode starts with 0 as in eofs
        PC = PC.assign_coords(mode=PC.mode+1)
        EOF = EOF.assign_coords(mode=EOF.mode+1)
    
    # just force PCs to normal distribution with unit variance
    def standardize(ds,dim="time"):
        return (ds-ds.mean(dim))/ds.std(dim)

    print(f'the first {nmode} modes explain {round(VarEx.sum("mode").values * 100,2)}% of the total variance of {da.name}')
    return standardize(PC), EOF
