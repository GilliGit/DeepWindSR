import numpy as np
from glob import glob

def proc_stoch(p0='/nfs/data/dahl0071/thesis/data/out/',model='wind-20220614-184509',dom=""):
    p0=p0+"/"+model
    files=glob(p0+"/stoch_{}*.npy".format(dom))
    print("found {}".format(len(files)))
    d={}
    dat=np.load(files[0])
    for i in range(dat.shape[0]):
        d[str(i)]=dat[i,:,:,:][np.newaxis,:,:,:]
    for file in files[1:]:
        dat=np.load(file)
        for i in range(dat.shape[0]):
            d[str(i)]=np.concatenate([d[str(i)],dat[i,:,:,:][np.newaxis,:,:,:]],axis=0)
    means={}
    var={}
    for key in d.keys():
        np.save(p0+"/ensemble_mean{0}_{1}.npy".format(dom,key),np.mean(d[key],axis=0))
        np.save(p0+"/ensemble_var{0}_{1}.npy".format(dom,key),np.var(d[key],axis=0))
proc_stoch(model="wind-20220618-010116",dom="CE")
proc_stoch(model="wind-20220618-010116",dom="GB")
