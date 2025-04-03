#!/usr/bin/env python3

import sys, os, argparse
import numpy as np
import xesmf as xe
import xarray as xr
from netCDF4 import Dataset


def create_regridder( fnin = "ocean_static.nc",
                      lat_name = "geolat",
                      lon_name = "geolon", 
                      res  = 0.25, 
                      fnout = "xemsf_weights.nc",
                      interp_method = "bilinear",
                      periodic = True ):
    f = Dataset(fnin, "r")
    lat_in = f.variables[lat_name][:].squeeze()
    lon_in = f.variables[lon_name][:].squeeze()
    f.close()

    lat_out = np.arange(-90.,  90.+res,res)
    lon_out = np.arange(-180.,180.,    res)

    grd_in  = {"lon": lon_in,  "lat": lat_in}
    grd_out = {"lon": lon_out, "lat": lat_out}

    regridder = xe.Regridder(grd_in, grd_out, interp_method, periodic=periodic)

    fntmp = f"{fnout}_step1"
    regridder.filename = fntmp
    regridder.to_netcdf()

    print(regridder)

    print("attach grid info into output file:", fnout)
    ds = xr.open_dataset(regridder.filename)
    ds.attrs["interp_method"] = interp_method
    ds.attrs["periodic"] = "yes" if periodic else "no"
    if grd_in['lon'].ndim == 2:
        ds["lat_grd_in"] = xr.DataArray(grd_in['lat'], dims=("lat_in","lon_in") )
        ds["lon_grd_in"] = xr.DataArray(grd_in['lon'], dims=("lat_in","lon_in") )
    elif lat_in.ndim == 1:
        ds["lat_grd_in"] = xr.DataArray(grd_in['lat'], dims=("lat_in") )
        ds["lon_grd_in"] = xr.DataArray(grd_in['lon'], dims=("lon_in") )

    if lat_out.ndim == 2:
        ds["lat_grd_out"] = xr.DataArray(grd_out['lat'], dims=("lat_out","lon_out") )
        ds["lon_grd_out"] = xr.DataArray(grd_out['lon'], dims=("lat_out","lon_out") )
    elif lat_out.ndim == 1:
        ds["lat_grd_out"] = xr.DataArray(grd_out['lat'], dims=("lat_out") )
        ds["lon_grd_out"] = xr.DataArray(grd_out['lon'], dims=("lon_out") )
    ds.to_netcdf(fnout)
    if os.path.exists(fntmp): os.remove(fntmp)


def load_regridder(fnin):

    f = Dataset(fnin,"r")
    f.set_auto_mask(False)
    grd_in  = {"lat": f.variables["lat_grd_in"][:], 
                "lon": f.variables["lon_grd_in"][:] }
    grd_out = {"lat": f.variables["lat_grd_out"][:],
                "lon": f.variables["lon_grd_out"][:] }
    interp_method = f.getncattr("interp_method") 
    periodic = True if f.getncattr("periodic") == "yes" else False
    f.close()
    regridder = xe.Regridder(grd_in, 
                             grd_out, 
                             interp_method, 
                             periodic=periodic,
                             weights=fnin)
    regridder.filename = fnin
    print("INFO of loaded regridder",regridder)

    return regridder


def parse_args():
    parser = argparse.ArgumentParser(description=("Generate regridder used by xesmf"))
    parser.add_argument("fnout", default="xemsf_wts_ufs0d25_LL0d25.nc", type=str, help=("name of output files"))
    parser.add_argument("--fngrd", default="ufs.cpld.cpl.r.2024-01-01-10800.nc", type=str, required=True, help=("Files storing ocean grid variables "))
    parser.add_argument("--latname", default="ocnExp_lat" ,type=str, required=True, help=("variable name of lat"))
    parser.add_argument("--lonname", default="ocnExp_lon" ,type=str, required=True, help=("variable name of lon"))
    parser.add_argument("--res", default=0.25, type=float, required=True, help=("resolution of the remapped latlon grids"))

    args = parser.parse_args()
    args.fngrd = os.path.abspath(args.fngrd)
    args.fnout = os.path.abspath(args.fnout)
    print(args)

    create_regridder( fnin = args.fngrd, 
                      lat_name = args.latname,
                      lon_name = args.lonname,
                      res = args.res,
                      fnout = args.fnout )
    regridder = load_regridder(args.fnout)


if __name__ == '__main__':
    print(" ".join(sys.argv[:]))
    parse_args()
    
 
