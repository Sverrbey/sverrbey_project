# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 14:24:08 2016

Script formerly known as DownloadThreddsMEThourTemp.py
@author: sarmar

Modified Sept 2016 by sjurk 
to import gridded hourly SeNorge temperature from MET Thredds

TODO: Modified March 2021 by Sjur.Kolberg@enki-hydrologi.no
to fit Python 3 and new thredds directory structure

"""

# Daily data from SeNorge2018 - version 20-05 - Latest or Archive
# At the HTTPTHREDDS site https://thredds.met.no/thredds/catalog/senorge:

# HTTPTHREDDS/seNorge_2018/Latest/catalog.html contain daily one-tstep files,
# currently (March 23, 2021) from 2020-12-22 to today. The files are 16-17 MB in size.

# HTTPTHREDDS/seNorge_2018/Archive/catalog.html contain annual ncfiles with daily
# time series, from 1957 to curent year. These are 2.2 GB in size.

# Hourly data from SeNorge2.0 - not corrected for wind-induced undercatch
# HTTPTHREDDS/seNorge2/provisional_archive/TEMP1h/gridded_dataset/202101/catalog.html
# contains hourly 12 MB files from Oct-2015 (TEMP) or Des-2015 (PREC) up to last hour.
# HTTPTHREDDS/seNorge2/archive/TEMP1h/catalog.html
# contains monthly files, 600-750 MB in size, from Jan-2010 to Dec-2016.


# import urllib
from calendar import monthrange
from datetime import datetime, timedelta
import requests
import os
import json
import traceback
from osgeo import gdal, osr, ogr, gdalconst
from netCDF4 import Dataset

if __name__ == "__main__":
    from .utils import ExportEnkiNcVar, SetGDALMetadata, GetNcInfo, MemDSfromInfo, Resample
else:
    from Tasks.utils import ExportEnkiNcVar, SetGDALMetadata, GetNcInfo, MemDSfromInfo, Resample

#https://thredds.met.no/thredds/fileServer/senorge/seNorge_2018/Latest/seNorge2018_20210329.nc
def DownloadSN2018days(start, end, local_rootfld):
    print (f"DownloadSN2018days begins with start = {start}", flush=True)
    if datetime.utcnow()-start > timedelta(days=30):
        return "Consider DownloadMultiyearSeries() or manual download for non-recent files"
    archive     = "https://thredds.met.no/thredds/fileServer/senorge/"
    urlfolder   = "seNorge_2018/Latest/"
    localfolder = f"{local_rootfld}{os.sep}seNorge_2018{os.sep}"
    if not os.path.isdir(f"{localfolder}"):
        return f"Missing folder {localfolder}"
    missing_files = 0
    downloaded_files = []

    stdname = { 'rr': 'daily_total_precipitation (06-06)', 
                'tg': 'daily_mean_temperature', 
                'tn': 'daily_min_temperature', 
                'tx':'daily_max_temperature'}
    units = {'rr': 'mm', 'tg': 'degC', 'tn': 'degC', 'tx':'degC'}
    flagcode = -999.99

    for i in range(0, (end-start).days+1):
        day = start + timedelta(days=i)
        fname = day.strftime("seNorge2018_%Y%m%d.nc")
        print ("DownloadSN2018days looks for "+fname, flush=True)
        if os.path.isfile(localfolder+fname): continue # time stamp, see TODO below
        url = archive + urlfolder + fname
        if __name__ == "__main__":
            print("url = ", url, flush=True)
            print("localfname = ", localfolder + fname, flush=True)

        try:
            res = requests.get(url, allow_redirects=True)
        except:
            missing_files += 1
        else:                           # The except-else block is only run if no exception is 
            if res.status_code == 200:  # thrown, but is not itself protected by the try clause.
                print ("DownloadSN2018days downloads "+url, flush=True)
                open(localfolder + fname, 'wb').write(res.content)
                downloaded_files.append(localfolder + fname)
                # In addition to today's file, add to this year's seNorge2018 database.
                # This can also be downloaded from (for 2021):
                # https://thredds.met.no/thredds/fileServer/senorge/seNorge_2018/Archive/seNorge2018_2021.nc
                year = day.year
                yearbasefname = localfolder + f"seNorge2018_{year}.nc"  # Filename also used at THREDDS

                for varname in ['rr', 'tg']:
                    SubDSfname = f'NETCDF:"{localfolder}{fname}":{varname}'  
                    print("SubDSfname = ", SubDSfname, flush=True)      # GDAL sees NcVariables as
                    DS = gdal.Open(SubDSfname)                          # subdatasets. Open separately.
                    if DS is not None:
                        SetGDALMetadata(DS, varname, stdname[varname], units[varname], flagcode)
                        ExportEnkiNcVar(DS, yearbasefname, day, timedelta(days=1), "seNorge")
            else: missing_files += 1

    if __name__ == "__main__":
        print (f"DownloadSN2018days ends with {len(downloaded_files)} downloaded files", flush=True)
    return downloaded_files



# DownloadHourSer fetches recent-years (currently 2016->) hourly data stored at 
# https://thredds.met.no/thredds/fileServer/senorge/seNorge2/provisional_archive/
# Variables are PREC1h (uncorrected) and TEMP1h, 
# At the URL, the data are in single-hour .nc-files in yearmonth folders
# Locally, files are stored in /seNorge2/TEMP1h (or PREC1h), with no period folder.
# To download longer periods, consider DownloadMultiyearSeries().
def DownloadHourSer(variable, start, end, local_rootfld):
    if variable not in ['PREC1h', 'TEMP1h']:
        return f"DownloadHourSer support only PREC1h amd TEMP1h, not {variable}"

    dataset = "seNorge2"
    if start.year<2016:
        return "Please use the DownloadMultiyearSeries() function for data prior to 2016"
    elif start > end:
        return "Start time before end time, please!"

    stdname = {'PREC1h': 'precipitation_amount', 'TEMP1h': 'air_temperature'}
    units = {'PREC1h': 'mm', 'TEMP1h': 'degC'}
    flagcode = -999.99

    # archive + urlfolder + /201901/seNorge_v2_0_PREC1h_grid_2019013123_2019013123.nc
    # archive + urlfolder + /202103/seNorge_v2_0_TEMP1h_grid_2021032908.nc
    archive     = "https://thredds.met.no/thredds/fileServer/senorge/"
    urlfolder   = f"seNorge2/provisional_archive/{variable}/gridded_dataset/"
    localfolder = f"{local_rootfld}{os.sep}{dataset}{os.sep}{variable}{os.sep}"
    if not os.path.isdir(f"{localfolder}"):
        return f"Please create subfolder {localfolder} before downloading to {local_rootfld}"
    missing_files = 0
    downloaded_files = []
    for year in range(start.year, end.year+1): 
        startmonth  = 1 if year > start.year else start.month
        endmonth    = 12 if year < end.year else end.month

        for month in range(startmonth,endmonth+1):
            starthour = 1 if year > start.year or month > start.month\
                        else (start.day-1) * 24 + start.hour
            endhour = monthrange(end.year, end.month)[1] * 24\
                        if year < end.year or month < end.month\
                        else (end.day-1) * 24 + end.hour + 1
            # nhours = endhour - starthour
            ttags = [(datetime(year,month,1) + timedelta(hours=h)).strftime("_%Y%m%d%H") \
                                                for h in list(range(starthour,endhour))]
            print(f"Working on {year}, {month}, (hour {starthour} to {endhour})", ttags, flush=True)
            for ttag in ttags:
                fname = f"seNorge_v2_0_PREC1h_grid{ttag}{ttag}.nc" if variable=="PREC1h" \
                    else f"seNorge_v2_0_TEMP1h_grid{ttag}.nc"
                if os.path.isfile(localfolder+fname): 
                    continue # time stamp, see TODO below
                url = archive + urlfolder + ttag[1:7] + '/' + fname
                print("url = ", url, flush=True)
                print("localfname = ", localfolder + fname, flush=True)

                try:
                    res = requests.get(url, allow_redirects=True)
                except:
                    missing_files += 1
                else:                           # The except-else block is only run if no exception is 
                    if res.status_code == 200:  # thrown, but is not itself protected by the try clause.
                        open(localfolder + fname, 'wb').write(res.content)
                        downloaded_files.append(localfolder + fname)
                        # In addition to hourly files, add to this month's seNorge2 database,
                        # with the same file name as in THREDDS seNorge2/archive (2010-2016)
                        dt = datetime.strptime(ttag, "_%Y%m%d%H")
                        monthbasefname = f"seNorge2_{variable}_grid_{year}{month}.nc"
                        DS = gdal.Open(localfolder + fname)     # Reopen the new NcFile with GDAL
                        if DS is not None:
                            SetGDALMetadata(DS, variable, stdname[variable], units[variable], flagcode)
                            ExportEnkiNcVar(DS, monthbasefname, dt, timedelta(hours=1),"seNorge")
                        else:
                            print(f"SeNorge::DownloadHourSer: Could not re-open {fname} for exporting to {monthbasefname}")
                        DS = None
                    else: missing_files += 1

    if __name__ == "__main__":
        print (f"Finished downloading, {missing_files} files were not found")

    return downloaded_files



def DownloadMultiyearSeries(variable, firstyear, lastyear, local_rootfld):
    if variable not in ['PREC1h', 'TEMP1h', 'PREC1d', 'TEMP1d', 'SN2018']:
        return f"DownloadMultiyearHourSeries does not support {variable}"\
                + "\nSupported variables are PREC1h, TEMP1h, PREC1d, TEMP1d, SN2018"

    dataset = "seNorge2" if variable.endswith("1h") else "seNorge_2018"

    if   dataset=="seNorge2" and firstyear<2010:
        return "The SeNorge 2.0 hourly data set does not include data prior to 2010"
    elif dataset=="seNorge_2018" and firstyear<1957:
        return "The seNorge_2018 hourly data set does not include data prior to 1957"
    elif firstyear>lastyear:
        return "First year before last year, please!"

    if not os.path.isdir(local_rootfld):#   = "D:"
        return f"Folder not found, must exist. Please create {local_rootfld}"


    archive = "https://thredds.met.no/thredds/fileServer/senorge/"

    # {archive}seNorge2/provisional_archive/PREC1h/gridded_dataset/201901/seNorge_v2_0_PREC1h_grid_2019013123_2019013123.nc
    # {archive}seNorge2/provisional_archive/PREC1h/201901/seNorge_v2_0_TEMP1h_grid_2019010101.nc
    # {archive}seNorge2/provisional_archive/TEMP1h/gridded_dataset/201701/seNorge_v2_0_TEMP1h_grid_2017012706.nc
    # {archive}seNorge2/archive/TEMP1h/seNorge2_TEMP1h_grid_201612.nc
    year = firstyear
    while year <= lastyear:

        if dataset=="seNorge2":  # Hourly data go in annual folders; monthly files
            # The local target folder must contain subfolders 2012/ etc

            varfolder = f"{local_rootfld}{os.sep}{dataset}{os.sep}{variable}{os.sep}"
            if not os.path.isdir(f"{varfolder}"):
                return f"Missing folder {varfolder}"
            urlfolder   = f"seNorge2/archive/{variable}/" if year <= 2016 \
                    else  f"seNorge2/provisional_archive/{variable}/gridded_dataset/"

            for month in range(1,13):
                if year <= 2016:
                    fname = f"seNorge2_{variable}_grid_{year}{month:02d}.nc"
                    url = archive + urlfolder + fname # No period folder at THREDDS, 
                    res = requests.get(url, allow_redirects=True) # just monthly ncfiles
                    if res.status_code == 200:
                        print (f"Downloaded {fname} as {res.headers['content-type']}")
                        open(f"{varfolder}{fname}", 'wb').write(res.content)
                else:
                    start = datetime(year, month, 1)
                    yrmonthfld = start.strftime("%Y%m") 
                    nhours = monthrange(year, month)[1] * 24
                    ttags = [(start + timedelta(hours=h)).strftime("_%Y%m%d%H") \
                                                for h in list(range(0,nhours,1))]
                    print("Working on ", year, month, flush=True)
                    for ttag in ttags:
                        fname = f"seNorge_v2_0_PREC1h_grid{ttag}{ttag}.nc" if variable=="PREC1h" \
                            else f"seNorge_v2_0_TEMP1h_grid{ttag}.nc"
                        url = archive + urlfolder + yrmonthfld + '/' + fname
                        #print("url = ", url, flush=True)
                        res = requests.get(url, allow_redirects=True)
                        if res.status_code == 200:
                            open(f"{varfolder}{yrmonthfld}{os.sep}{fname}", 'wb').write(res.content)
                            print(f"Saved as {varfolder}{yrmonthfld}{os.sep}{fname}", flush=True)
                        else: print("Could not download url = ", url, flush=True)
                        #else:
                        #    print("Status code = ", res.status_code)
                        #break
        else:   # SN2018 daily data go directly in varfolder; these are yearly files
            localfolder = f"{local_rootfld}{os.sep}{dataset}"
            if not os.path.isdir(localfolder):
                return f"Please create folder {localfolder}"
            fname = f"seNorge2018_{year}.nc"
            url = f"{archive}seNorge_2018/Archive/{fname}"
            res = requests.get(url, allow_redirects=True)
            if res.status_code == 200:
                open(f"{localfolder}{os.sep}{fname}", 'wb').write(res.content)
                print (f"Downloaded {url} as {res.headers['content-type']}\
                        to {localfolder}{os.sep}{fname}")

        year += 1

    if __name__ == "__main__":
        print ("Finished!")



# From a temporal stack of seNorge ncfiles residing in datafld, create a 
# multi-band memory GDALDataset for each variable.
# For TEMP1h and PREC1h, obsfilelist has flag entries for missing files, and
#   maps are interpolated temporally to make the time series even and complete.
# Return a list of one or more multi-band GDALDataset sets:
#   TEMP1h and PREC1h:  A single GDALdataset with gaps interpolated
#   SeNorge2018:        Two GDALDatasets (P, T) with gaps interpolated
def CreateTemporalGDALstack(datafld, obsfilelist):
    if not datafld.endswith(os.sep):
        datafld += os.sep                   

    if not os.path.isdir(datafld):
        return None, 'CreateTemporalGDALstack: Could not find folder ' + datafld 
    if type(obsfilelist) is not list or len(obsfilelist) < 1:
        return None, 'CreateTemporalGDALstack: Invalid type or length of obsfilelist'
    if not os.path.isfile(datafld + obsfilelist[0]):
        return None, f'CreateTemporalGDALstack: {obsfilelist[0]} not found in {datafld}'
    nfiles = len(obsfilelist)

    # Get some time info from the file names

    if "seNorge_2018" in datafld:
        dt, dtsub, frmt = timedelta(days=1), slice(-11,-3), "%Y%m%d"
    else:
        dt, dtsub, frmt = timedelta(hours=1), slice(-13,-3), "%Y%m%d%H"
    firstdt = datetime.strptime(obsfilelist[0][dtsub],frmt)
    lastdt = datetime.strptime(obsfilelist[-1][dtsub],frmt)
    nbands = int((lastdt-firstdt).total_seconds() / dt.total_seconds()) + 1
    # nbands is the full number of entries from start to end, including where files are missing.

    testncfname = datafld + obsfilelist[0]
    ncfileinfo = GetNcInfo(testncfname)     # GetNcInfo reads metadata and
    nt = ncfileinfo['nt']                   # closes the file after use. These
    nRows = ncfileinfo['ny']                # metadata should be the same for
    nCols = ncfileinfo['nx']                # all files in obsfilelist.
    
    wktstring = ncfileinfo['wkt'] if 'wkt' in ncfileinfo else ""
    projstring = ncfileinfo['proj4'] if 'proj4' in ncfileinfo else ""
    assert wktstring != "" or projstring != "", f"SeNorge::CreateTemporalGDALstack: {testncfname} appears to lack coordsystem info"

    xylim = [ncfileinfo['xmin'], ncfileinfo['xmax'],\
             ncfileinfo['ymax'], ncfileinfo['ymin']] # WENS
    mapvarnames = ncfileinfo['mapvars']     # metfields, lat/longitude, elevation

    if nfiles == 1: nbands = int(nt)
    else:           assert nt==1

    SrcNcDS = Dataset(testncfname)          # Again open testncfname to get info
    ncv = SrcNcDS.variables                 # on each individual variable for 
    MemDataSets = []                        # creating GDALDatasets

    flag = -999.99                          # Will be updated

    for varname in mapvarnames:             # Construct a GDALDataset for each
        mapvar = ncv[varname]               # metfield variable in the ncfile
        if varname not in ["precipitation_amount", "temperature", "rr", "tg"]:
            continue
        newDS = MemDSfromInfo(nRows, nCols, nbands, xylim, wktstring, projstring)
        assert newDS != None, f"Could not construct GDALDataset {varname} from {nRows}, {nCols}, {nbands}, {xylim}, {wktstring}, {projstring}"
        MemDataSets.append(newDS)
        stdname = mapvar.__dict__['standard_name']
        units = mapvar.__dict__['units']
        flag = float(mapvar.__dict__['_FillValue'])
        SetGDALMetadata(MemDataSets[-1], varname, stdname, units, flag)

    SrcNcDS.close() 
    lastgoodidx = -1

    # We now know bandno iterates over even time steps, and nbands is the expected number
    # of files, if all are present. Most of this block is to handle temporal interpolation
    for tidx, fname in enumerate(obsfilelist):
        if fname.startswith("Gap from: "):  # Detect missing files
            if lastgoodidx == -1:
                lastgoodidx = tidx-1        # Found the start of a gap
            continue                        # Do nothing until the gap is closed

        fullname = datafld + fname          
        assert os.path.isfile(fullname),    f"SeNorge: File {fullname} should have existed"
        try:
            SrcNcDS = Dataset(fullname)
        except:                             # File may be corrupt, pretend it is missing.
            print (f"SeNorge: File {fullname} could not be opened, treated as missing", flush=True)
            if lastgoodidx == -1:
                lastgoodidx = tidx-1        # Found the start of a gap
            continue                        # Do nothing until the gap is closed



        ncv = SrcNcDS.variables

        if lastgoodidx >= 0:                # This file closes a gap, fill up all
            for MemDS in MemDataSets:       # the missed tsteps by interpolation
                varname = MemDS.GetMetadata()['name']
                ncvar = ncv[varname]
                lastgoodmap = MemDS.GetRasterBand(lastgoodidx+1).ReadAsArray()
                for tmptidx in range(lastgoodidx+1,tidx,1):
                    lastgoodwgt = (tidx-tmptidx) / (tidx-lastgoodidx)
                    nextgoodwgt = (tmptidx-lastgoodidx) / (tidx-lastgoodidx)
                    rband = MemDS.GetRasterBand(tmptidx+1)
                    rband.SetNoDataValue(flag)
                    rband.WriteArray(   lastgoodmap  * lastgoodwgt \
                                      + ncvar[0,:,:] * nextgoodwgt    )
            lastgoodidx = -1
            # End of dataset loop
        for MemDS in MemDataSets:           # Write the data array for this time step
            varname = MemDS.GetMetadata()['name']
            if not varname in ncv:
                print (f"Variable {varname} not found in NcFile {fullname}, skipping", flush=True)
                continue
            ncvar = ncv[varname]
            rband = MemDS.GetRasterBand(tidx+1)
            rband.SetNoDataValue(flag)
            rband.WriteArray(ncvar[0,:,:])
        # End of dataset loop
    # End of tidx / file list loop  

    # Before we return, rename the variables to distinguish better from Arome variables.
    for MemDS in MemDataSets:
        varname = MemDS.GetMetadata()['name']
        if   varname == "precipitation_amount": newname = "PREC1h"  # SeNorge 2.0 hourly prec
        elif varname == "temperature": newname = "TEMP1h"           # SeNorge 2.0 hourly temp
        elif varname in ["rr","tg"]: newname = varname              # SeNorge2018 rr or tg
        MemDS.SetMetadataItem('name',newname)

    return MemDataSets, firstdt




def GetNewData(dataset, storagefld, start=None, end=None):

    if not dataset in ['TEMP1h', 'PREC1h', 'SN2018']:
        return "Usage: GetNewData(dataset, regionfld, start=None, end=None)\n \
                        with one of 'TEMP1h', 'PREC1h', 'SN2018' as dataset"
    if not os.path.isdir(storagefld):
        return f"Missing folder: {storagefld}"
    if not storagefld.endswith(os.sep):    
        storagefld += os.sep

    if end==None:
        end = datetime.utcnow()
    if start==None:
        start = datetime.utcnow() - timedelta(hours=120)
    period = end - start
    if period.total_seconds() < 0:
        return f"GetNewData() for {dataset} failed due to start-end inconsistency"

    newfiles = []
    if dataset == 'SN2018':
        newfiles = DownloadSN2018days(start.replace(hour=6,minute=0,second=0,microsecond=0), 
                                        end, storagefld)
    else:
        newfiles = DownloadHourSer(dataset, start, end, storagefld)
    numnewfiles = len(newfiles)

    if numnewfiles > 1:
        fspec = f" (from {newfiles[0]} to {newfiles[-1]})."
    elif numnewfiles == 1:
        fspec = f" ({newfiles[0]})."
    else:
        fspec = "."
    
    return f"{numnewfiles} new {dataset} files downloaded{fspec}\n"




    # Return a list of <dataset> files in regionfld/dataset/, for the specified
    # period. A None in fromtime or totime means no limit in that direction.
    # For single-time PREC1h and TEMP1h files, any missing file is represented
    # by a flag string, keeping the list at fixed time interval. The first
    # element in the returned list is always an existing file, thus the
    # returned list is not guaranteed to start at fromtime.
def GetFileList(product, datafld, fromtime, totime):
    if not datafld.endswith(os.sep):
        datafld += os.sep

    regionfld = datafld[0:datafld[0:-1].rfind(os.sep)+1]

    if product == "TEMP1h":         # seNorge_v2_0_TEMP1h_grid_2021021616.nc
        fileprefix = "seNorge_v2_0_TEMP1h_grid_"
        dtsub, frmt = slice(25,35), "%Y%m%d%H"
    elif product == "PREC1h":       # seNorge_v2_0_PREC1h_grid_2021021616_2021021616.nc
        fileprefix = "seNorge_v2_0_PREC1h_grid_"
        dtsub, frmt = slice(25,35), "%Y%m%d%H"
    elif product == "SN2018":       # seNorge2018_20210222.nc
        fileprefix = "seNorge2018_"
        dtsub, frmt = slice(12,20), "%Y%m%d"
    else:   return f"Unsupported subfolder {datafld} (product {product})\
                     supplied to SeNorge::GetFileList()"

    # The len(fn > 20) trick serves to exclude the monthly seNorge2018_2021.nc
    filelist = [fn for fn in os.listdir(datafld) if fn.startswith(fileprefix) and len(fn)>20] 
    filelist.sort()                 # Puts files in temporal order

    SrcTstep = timedelta(minutes=60)
    prevTime = None
    hrfilelist = list()
                            # Copy file names from filelist to obsfilelist.
    for zitem in filelist:  # Insert a "Gap from..." string for any missing
                            # file when product is 'PREC1h' or 'TEMP1h'.
        now = datetime.strptime(zitem[dtsub],frmt)
        if fromtime is not None and now < fromtime:  continue
        if totime is not None and now > totime:    break

        if product == "SN2018":
            hrfilelist.append(zitem)   
            continue                    # Skip temporal interpolation for SeNorge2018

        if prevTime==None:              # First temporal file
            prevTime = now - SrcTstep
            assert(len(hrfilelist)==0)
        else:                           # Not the first file, prevTime is valid
            assert(len(hrfilelist)>0)
            dt = now - prevTime         # Validate time sequence of files
            if dt > SrcTstep:           # Time interval longer than expected
                flag = "Gap from: " + prevTime.strftime("%d.%m.%Y %H:%M")
                flag +=     " to: " + now.strftime("%d.%m.%Y %H:%M")
                # print (now, dt, SrcTstep, zitem, flag)

            while dt > SrcTstep:        # For each missing file, insert the 
                hrfilelist.append(flag) # "Gap from ..." message in hrfilelist
                prevTime += SrcTstep    # where the missing file name would have 
                dt = now - prevTime     # been, so hrfilelist remains hourly spaced.

        hrfilelist.append(zitem)    # The only real file name added in this iter.
        prevTime = now              # Time of the file just read, always existing.
        
    return hrfilelist





def ResampleExport(dataset, mainfolder, regiondirs, fromtime=None, totime=None):
    # ResampleExport is called from the scheduler.
    # Calls GetFileList to get a list of locally available 1-hr files.
    # Builds a temporal MEM GDALDataset from these, and geotransforms each
    # band into a new MEM GDALDataset, which is exported to the NetCDF
    # file supplied as ENKIbasefname. Returns logmsg.
    logmsg = ""
    try:
        if not mainfolder.endswith(os.sep):
            mainfolder += os.sep
        if not dataset in ['SN2018', 'TEMP1h', 'PREC1h']:
            return "Usage: ResampleExport(dataset, mainfolder, NcBasefname = "") with:\n \
                one of 'TEMP1h', 'PREC1h', 'SN2018' as dataset"

        datafld = f"{mainfolder}seNorge_2018" if dataset == "SN2018"\
            else  f"{mainfolder}seNorge2{os.sep}{dataset}"

        localfilelist = GetFileList(dataset, datafld, fromtime, totime) 

        # From the list of files, Create a time-gap-filled GDALDataset from the 
        # files returned by GetFileList
        # TEMP1h and PREC1h result in one GDALDataset holding the time series.
        # SeNorge2018 results in two GDALDatasets (prec/temp) holding time series

        if __name__ == "__main__":
            print(f"Calling CreateTemporalGDALstack with a list of {len(localfilelist)} files:")
        WndTserDS, startTime = CreateTemporalGDALstack(datafld, localfilelist)

        logmsg += f"{datetime.utcnow().strftime('%Y.%m.%d %H:%M')}: "
        if WndTserDS==None or len(WndTserDS)==0:
            logmsg += f"No GDAL stack created for {dataset} at {startTime}\n"    
        else:
            logmsg += f"{len(WndTserDS)} temporal GDAL stacks created from {len(localfilelist)} "\
                    + f"{dataset} files in {datafld} (last file {localfilelist[-1]})\n"


        regionhome = f"{mainfolder}Regions{os.sep}"

        for regfld in regiondirs:
            regionfld   = f"{regionhome}{regfld}{os.sep}"
            ENKIbase    = f"{regionfld}HourlyForecastInput.nc"
            DayBase     = f"{regionfld}SeNorge2018_PT.nc"

            with open(regionfld+'RegionConfig.json') as f:
                Cfg = json.load(f)
            tmpl_rst_fname = regionfld + Cfg["elev_GIS_dataset"]

            # if len(WndTserDS) > 0:
            #     logmsg += f"{dataset}: Resampling {len(WndTserDS)} data sets for region {regfld}:\n"
            for origDS in WndTserDS:        # A single file for PREC1h/TEMP1h, two for SN2018.
                regDS = Resample(origDS, tmpl_rst_fname)
                logmsg += f"{datetime.utcnow().strftime('%Y.%m.%d %H:%M')}: "
                if dataset == 'SN2018':
                    ExportEnkiNcVar(regDS, DayBase, startTime, timedelta(days=1),"seNorge")
                    logmsg += f"{origDS.GetMetadata()['name']} resampled and exported to {DayBase}\n"
                else:
                    ExportEnkiNcVar(regDS, ENKIbase, startTime, timedelta(hours=1),"excel")
                    logmsg += f"{origDS.GetMetadata()['name']} resampled and exported to {ENKIbase}\n"
    except Exception as e:
        msg = format("%s: %s") % (type(e),str(e))
        logmsg += f"SeNorge::ResampleExport raised exception: {msg}"
        traceback.print_exc()
        
    return logmsg





if __name__ == "__main__":

    mainfolder = 'G:\\Robot\\'

    logmsg = GetNewData('TEMP1h', mainfolder)
    print(logmsg)
    logmsg = GetNewData('PREC1h', mainfolder)
    print(logmsg)
    logmsg = GetNewData('SN2018', mainfolder)
    print(logmsg)

    regionhome = f"{mainfolder}Regions{os.sep}"
    regiondirs = [d for d in os.listdir(regionhome) if os.path.isdir(f"{regionhome}{d}")] 

    logmsg = ResampleExport('TEMP1h', mainfolder, regiondirs)
    print(logmsg)
    logmsg = ResampleExport('PREC1h', mainfolder, regiondirs)
    print(logmsg)
    logmsg = ResampleExport('SN2018', mainfolder, regiondirs)
    print(logmsg)
