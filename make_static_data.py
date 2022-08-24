# get static data
assert files[0][-5:]!="SP.nc", "Error: first ncfile is spinup"
ds=xr.open_dataset(files[0],
                   drop_variables=vars_).isel(south_north=slice(2,412))
static_LR=ds[["HGT","LANDMASK"]].coarsen(south_north=10,west_east=10).mean()
static_LR["LANDMASK"]=static_LR.LANDMASK.round()
static_LR=np.transpose(static_LR.to_array().values,(1, 2, 0))

static_HR=np.transpose(ds[["HGT","LANDMASK"]].to_array().values,(1, 2, 0))

mu=np.mean(static_HR[:,:,0])
sig=np.std(static_HR[:,:,0])
static_HR[:,:,0]=(static_HR[:,:,0]-mu)/sig

mu=np.mean(static_LR[:,:,0])
sig=np.std(static_LR[:,:,0])
static_LR[:,:,0]=(static_LR[:,:,0]-mu)/sig

#close ds
ds.close()


Dom="CE"
p2s="/".join(["/nfs/data/dahl0071/thesis/data/static",Dom])
if not os.path.exists(p2s):
    os.makedirs(p2s)
np.save("/".join([p2s,"LR.npy"]), static_LR)
np.save("/".join([p2s,"HR.npy"]), static_HR)

