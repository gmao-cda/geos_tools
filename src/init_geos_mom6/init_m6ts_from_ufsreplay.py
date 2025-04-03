#!/usr/bin/env python3

import sys, os, numpy as np, datetime as dt
import argparse
import xesmf as xe
from netCDF4 import Dataset
import multiprocessing
from multiprocessing import shared_memory
from numba import jit
from regrid_tools import iterative_fill_POP_core, iterative_fill_sor

missing_pts_ic_value = 1.e10

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

    return regridder, grd_in, grd_out

def load_ufsocn_diag(fnDiag = "ocn_2021_05_01_00.nc"):
    f = Dataset(fnDiag, "r")
    #f.set_auto_mask(False)
    z1d    = f.variables["z_l"][:]
    t3dTri = f.variables["temp"][:].squeeze()
    s3dTri = f.variables["so"][:].squeeze()
    deltaHours = int(f.variables["time"][0])
    date0 = dt.datetime.strptime(f.variables["time"].units, "hours since %Y-%m-%d %H:%M:%S")
    odate = date0 + dt.timedelta(hours=deltaHours)
    f.close()

    return z1d, t3dTri, s3dTri, odate


def flood_level(ilev, shm_name, shape, use_fill_pop=False,tol=1.e-2,nitermax=100):
    shm = shared_memory.SharedMemory(name=shm_name)
    t_f = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    t_f_lev = t_f[ilev,:,:]
    ## flood land and -90 -> -80
    pts_to_fill = np.isnan(t_f_lev)

    use_fill_pop = False
    if use_fill_pop:
        t_f_lev[pts_to_fill] = missing_pts_ic_value
        iterative_fill_POP_core(var           = t_f_lev,
                                fillmask      = pts_to_fill,
                                missing_value = missing_pts_ic_value,
                                tol           = tol, ltripole = False, nitermax = nitermax)
    else:
        iterative_fill_sor(nlat = t_f_lev.shape[0],
                           nlon = t_f_lev.shape[1],
                           var  = t_f_lev,
                           fillmask = pts_to_fill,
                           tol = tol, rc = 1.5, ltripole = False, max_iter = nitermax)

    t_f_lev[0,:]  = t_f_lev[1,:].copy() # set South-most row
    print(f"finish flooding at lev={ilev}: min, max=",t_f_lev.min(),t_f_lev.max())
    shm.close()


def interp_flood_globle(regridder, t3dTri, s3dTri, npes = 8):

    # interp 
    t_f = regridder(t3dTri.filled(np.nan))
    s_f = regridder(s3dTri.filled(np.nan))
    print("AFTER interp: t_range=", np.nanmin(t_f),np.nanmax(t_f))
    print("AFTER interp: s_range=", np.nanmin(s_f),np.nanmax(s_f))

    # flood
    flood_t = True
    flood_s = True
    use_fill_pop = False

    shm = shared_memory.SharedMemory(create=True, size = t_f.size * 8) #size in bytes
    var_f_shared = np.ndarray(t_f.shape,dtype=np.float64, buffer=shm.buf)

    if flood_t:
       print("="*80+"\nFlood T")
       var_f_shared[:] = t_f
       tol = 5.e-3; nitermax = 2000
       with multiprocessing.Pool(processes=npes) as pool:
            pool.starmap(flood_level, [ (ilev, shm.name, t_f.shape, use_fill_pop, tol, nitermax) for ilev in range(t_f.shape[0]) ]   )
       t_f[:] = var_f_shared
       #t_f[-1,:,:] = missing_pts_ic_value   # deepest level has no valid value

    if flood_s:
       print("="*80+"\nFlood S")
       var_f_shared[:] = s_f
       tol = 5.e-3; nitermax = 2000
       with multiprocessing.Pool(processes=npes) as pool:
            pool.starmap(flood_level, [ (ilev, shm.name, s_f.shape, use_fill_pop, tol, nitermax) for ilev in range(s_f.shape[0]) ]   )
       s_f[:] = var_f_shared
       #s_f[-1,:,:] = missing_pts_ic_value   # deepest level has no valid value

    shm.close()
    shm.unlink()

    print("AFTER flooding: t_range=", np.min(t_f),np.max(t_f))
    print("AFTER flooding: s_range=", np.min(s_f),np.max(s_f))
    
    return t_f, s_f
    
def write_netcdf(fnout, lon1d, lat1d, z1d, t, s, odate):

    nz, ny, nx = t.shape
    t = t.reshape((1,nz,ny,nx))
    s = s.reshape((1,nz,ny,nx))

    f = Dataset(fnout,"w")
    # create dim
    d_time  = f.createDimension("time",  None)
    d_lev   = f.createDimension("depth", z1d.size)
    d_lat   = f.createDimension("lat",   lat1d.size)
    d_lon   = f.createDimension("lon",   lon1d.size)

    # create vars
    v_time  = f.createVariable("time", "f8", ("time") )
    v_depth = f.createVariable("depth","f8", ("depth") )
    v_lat   = f.createVariable("lat",  "f8", ("lat") )
    v_lon   = f.createVariable("lon",  "f8", ("lon") )

    # CDA: MOM6 uses _FillValue as missing_value to flag out grids in the Z file
    v_s   = f.createVariable("salt", "f4", ("time","depth", "lat", "lon"),fill_value = missing_pts_ic_value )
    v_t   = f.createVariable("temp", "f4", ("time","depth", "lat", "lon"),fill_value = missing_pts_ic_value )

    # add attributes
    v_time.units = odate.strftime("days since %Y-%m-%d %H:00:00")
    v_time.calendar = "gregorian" # from cmems time attribute
    v_time.cartesian_axis = "T"

    v_depth.units = "m"
    v_depth.direction = np.int32(-1)
    v_depth.cartesian_axis = "Z"

    v_lat.units = "degrees_north"
    v_lat.cartesian_axis = "Y"

    v_lon.units = "degrees_east"
    v_lon.cartesian_axis = "X"

    # fill in values
    v_time[:]   = 0.0
    v_depth[:]  = z1d
    v_lat[:]    = lat1d
    v_lon[:]    = lon1d
    v_s[:]      = s.astype(np.float32)
    v_t[:]      = t.astype(np.float32)
    f.close()

def parse_args():
    parser = argparse.ArgumentParser(description=("create T/S on uniform latlon grid with input of UFS replay ocean history files"))
    parser.add_argument("fnout", default="IC_TS_ufs.nc", type=str, help=("output T/S file to initialize MOM6"))
    parser.add_argument("--fnDiag",default="ocn_2016_01_02_00.nc", required=True, type=str,help=("UFS ocean diag files"))
    parser.add_argument("--fnRgrdr",default="xemsf_wts_ufs0d25_LL0d25.nc",required=True, type=str,help=("weights file used by regridder"))
    parser.add_argument("--npes", default=8, type=int, required=False, help=("num of procceses for parallel-processing"))

    args = parser.parse_args()
    args.fnDiag  = os.path.abspath(args.fnDiag)
    args.fnRgrdr = os.path.abspath(args.fnRgrdr)
    args.fnout   = os.path.abspath(args.fnout)
    print(args)

    regridder, grd_in, grd_out = load_regridder(args.fnRgrdr)
    z1d, t3dTri, s3dTri, odate = load_ufsocn_diag(args.fnDiag)
    t3d_f, s3d_f = interp_flood_globle(regridder, t3dTri, s3dTri, npes = 8)
    write_netcdf(args.fnout, grd_out['lon'], grd_out['lat'], z1d, t3d_f, s3d_f, odate)

if __name__ == '__main__':
    print(" ".join(sys.argv[:]))
    parse_args()



