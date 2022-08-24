import xarray as xr
from glob import glob
from utils import *
import os
from PatchCutter import PatchCutter

def make_train_data(ncfile,
                    p0="/nfs/data/dahl0071/thesis/data/",
                    vars2drop=[],
                    idx_times=np.arange(0,48,6),
                    data_type="train",
                    year="xxxx",
                    dom="CE",
                    patcher_a=None,
                    patcher_b=None
                   ):
    #idx_times: index's of selected times, default sample every 3h
    
    assert data_type in ["test","train","valid"], "data_type must be train, valid or test"
    assert dom in ["CE","GB"], "dom must be CE or GB"
    
    p0="/".join([p0,data_type,dom,year])
    if not os.path.exists(p0):
        os.makedirs(p0)   
    p0+="/"
    ncfilename=ncfile.split("/")[-1].split(".")[0]
    filename=p0+ncfilename+".tfrecord"
    
    # NOT take spin-up files
    if ncfile[-5:]!="SP.nc":
        ds=xr.open_dataset(ncfile,
                          drop_variables=vars2drop)
        #select 100m height
        ds=ds.isel(height=2)
        
        #select times (domain south-north extent also reduced to be compatible to 10x10 avg pooling filter)
        ds=ds.isel(time=idx_times)
        if dom=="CE":
            ds=ds.isel(south_north=slice(2,412))
        else:
            ds=ds.isel(west_east=slice(3,393))
        # drop timestamp if contains na in its field (very time consuming if vars2drop=[])
        ds=ds.dropna("time")
        
        if ds.dims["time"] ==0:
            print("No valid time steps for "+ncfilename)
            ds.close()
        else:
            #converting WS and WD to u,v components
            u=-ds.WS*np.sin(ds.WD*np.pi/180)
            v=-ds.WS*np.cos(ds.WD*np.pi/180)

            #close ds
            ds.close()

            #merge into new dataset
            u.name="u"
            v.name="v"
            uv=xr.merge([u,v]).to_array()

            if data_type=="test":
                filename_npy=p0+ncfilename+".npy"
                np.save(filename_npy,np.transpose(uv.values,(1, 2, 3, 0)))
                ro=None
            else:
                #patch cutting for training
                patcher_a.randomize()
                patcher_b.synchronize(patcher_a)
                ro=patcher_a.relative_offset
                uv,_=patcher_b(uv)

            #(C,N,H,W)-> (N,H,W,C)
            uv=np.transpose(uv.values,(1, 2, 3, 0))

            generate_TFRecords(filename,
                               uv, 
                               ro=ro,
                               mode=data_type, 
                               K=10)
            print("generated tfrecord for "+ncfilename)

            no_na=len(idx_times)-np.shape(uv)[0]
            if no_na!=0:
                print("{} timestamps dropped \n".format(no_na))

# vars to drop
vars_=['crs',
       'ABLAT_CYL',
       'ACCRE_CYL',
       'ALPHA',
       'HFX',
       'HGT',
       'LANDMASK',
       'LH',
       'LU_INDEX',
       'PBLH',
       'PD',
       'PRECIP',
       'PSFC',
       'Q2',
       'QVAPOR',
       'RHO',
       'RMOL',
       'SEAICE',
       'SWDDIR',
       'SWDDNI',
       'T',
       'T2',
       'TKE',
       'TSK',
       'UST',
       'WD10',
       'WS10',
       'ZNT']

years=np.arange(2011,2019,1)
#year="2017"
data_type="train"
domain="GB"
for year in years:
    year=str(year)
    files=glob("/nfs/group/fw/newa/production/"+domain+"/"+year+"/post_files/*")
    if data_type=="valid":
        times=np.arange(0,48,12)
    else:
        times=np.arange(0,48,6)
    if domain=="CE":
        Nsn,Nwe=24,32 #28,36
    else:
        Nsn,Nwe=32,24
    patch_a = PatchCutter(patch_size=(Nsn,Nwe))
    patch_b = PatchCutter(patch_size=(Nsn*10,Nwe*10))
    
    
    for file_ in files:
        print("starting on "+file_)
        make_train_data(ncfile=file_,
                        vars2drop=vars_,
                        idx_times=times,
                        data_type=data_type,
                        year=year,
                        dom=domain,
                        patcher_a=patch_a,
                        patcher_b=patch_b
                        )
    print("Finished year {} for {} domain".format(year,domain))

