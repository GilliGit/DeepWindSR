#!/bin/python
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('agg') #disable tk backend
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
#tf.disable_v2_behavior()
from scipy.interpolate import interp2d#,RBFInterpolator
import json
from NumpyEncoder import NumpyEncoder


def downscale_image(x, K):
    tf.reset_default_graph()

    if x.ndim == 3:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

    x_in = tf.placeholder(tf.float64, [None, x.shape[1], x.shape[2], x.shape[3]])
    weight = tf.constant(1.0/K**2, shape=[K, K, x.shape[3], 1], dtype=tf.float64)
    p_filt=tf.eye(2,batch_shape=[1,1],  dtype=tf.float64)
    downscaled = tf.nn.separable_conv2d(x_in,depthwise_filter=weight,pointwise_filter=p_filt, strides=[1, K, K, 1], padding='SAME')
    with tf.Session() as sess:
        ds_out = sess.run(downscaled, feed_dict={x_in: x})
    return ds_out


def plot_SR_HR_data(d, path):
    keys=list(d.keys())
    
    HR=d["HR"]
    keys.remove('HR')
    if "LR" in keys:
        LR=d["LR"]
        keys.remove('LR')
    else:
        LR=downscale_image(HR,10)
    
    Ncol=len(keys)+2
    for i in range(HR.shape[0]):
        #plt.figure(figsize=(16, 8))
        fig, ax = plt.subplots(2, Ncol,figsize=[16,6],gridspec_kw = {'wspace':0.05, 'hspace':0.05, "top":0.95,"bottom":0.02})
        #plotting limits based on HR data
        vmin0, vmax0 = np.min(HR[i,:,:,0]), np.max(HR[i,:,:,0])
        vmin1, vmax1 = np.min(HR[i,:,:,1]), np.max(HR[i,:,:,1])
        
        # HR leftmost plot
        ax_=ax[0,0]
        ax_.imshow(HR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower',aspect='auto')
        ax_.set_title('HR', fontsize=12)
        ax_.set_ylabel('u', fontsize=9)
        #plt.colorbar()
        ax_.set_xticks([], [])
        ax_.set_yticks([], [])
        #plt.tight_layout()
        
        ax_=ax[1,0]
        ax_.imshow(HR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower',aspect='auto')
        ax_.set_ylabel('v', fontsize=9)
        #plt.colorbar()
        ax_.set_xticks([], [])
        ax_.set_yticks([], [])
        #plt.tight_layout()
        
        # LR rightmost plot
        ax_=ax[0,-1]
        ax_.imshow(LR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower',aspect='auto')
        ax_.set_title('LR', fontsize=12)
        #plt.colorbar()
        ax_.set_xticks([], [])
        ax_.set_yticks([], [])
        #plt.tight_layout()
        
        
        ax_=ax[1,-1]
        ax_.imshow(LR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower',aspect='auto')
        #plt.colorbar()
        ax_.set_xticks([], [])
        ax_.set_yticks([], [])
        #plt.tight_layout()
        
        # loop through comparison models
        for j,key in enumerate(keys):
            SR=d[key]

            ax_=ax[0,j+1]
            im0=ax_.imshow(SR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower',aspect='auto')
            ax_.set_title(key, fontsize=12)
            #plt.colorbar()
            ax_.set_xticks([], [])
            ax_.set_yticks([], [])
            #plt.tight_layout()

            ax_=ax[1,j+1]
            im1=ax_.imshow(SR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower',aspect='auto')
            #plt.colorbar()
            ax_.set_xticks([], [])
            ax_.set_yticks([], [])
            #plt.tight_layout()
        #plt.tight_layout()
        #fig.tight_layout()
        #fig.subplots_adjust(right=0.8)
        fig.subplots_adjust(right=0.95)
        cbar_ax0 = fig.add_axes([0.96, 0.52, 0.03, 0.4])
        fig.colorbar(im0, cax=cbar_ax0)
        cbar_ax0.set_label('m/s')
        cbar_ax1 = fig.add_axes([0.96, 0.05, 0.03, 0.4])
        fig.colorbar(im1, cax=cbar_ax1)
        plt.savefig(path+'_HRvsSRimg{0:05d}.png'.format(i), dpi=200, bbox_inches='tight')
        plt.close()

def get_bicubic_interp_SR(grid,LR_val):
    # grid: dict with south_north, west_east wrf grid info (LR and HR)
    SR_all=None
    for i in range(np.shape(LR_val)[0]):
        fu=interp2d(grid["we_lr"],grid["sn_lr"],LR_val[i,:,:,0],kind="cubic")
        fv=interp2d(grid["we_lr"],grid["sn_lr"],LR_val[i,:,:,1],kind="cubic")
        zu = fu(grid["we"], grid["sn"])
        zv = fv(grid["we"], grid["sn"])

        SR=np.stack([zu,zv],axis=-1)
        SR=SR[np.newaxis,...]
        if SR_all is None:
            SR_all=SR
        else:
            SR_all=np.concatenate([SR_all,SR],axis=0)
    return SR_all

def get_RBF_interp_SR(LR_val,xobs,xflat,N_sn,N_we):
    SR_all=None
    for i in range(np.shape(LR_val)[0]):
        uobs=np.ravel(LR_val[i,:,:,0])
        vobs=np.ravel(LR_val[i,:,:,1])

        uflat = RBFInterpolator(xobs, uobs, smoothing=0, kernel='cubic')(xflat)
        ugrid = uflat.reshape(N_sn,N_we)

        vflat = RBFInterpolator(xobs, vobs, smoothing=0, kernel='cubic')(xflat)
        vgrid = vflat.reshape(N_sn,N_we)

        SR=np.stack([ugrid,vgrid],axis=-1)
        SR=SR[np.newaxis,...]
        if SR_all is None:
            SR_all=SR
        else:
            SR_all=np.concatenate([SR_all,SR],axis=0)
    return SR_all


def get_metrics(d,d2,d_results,HR_norm,id_):
    dif=d_results["HR"]-d_results[id_]
    SR_norm=np.linalg.norm(d_results[id_],axis=-1)
    dif_mag=HR_norm-SR_norm
    dif_cos=np.sum(d_results["HR"]*d_results[id_],axis=-1)/(HR_norm*SR_norm)
    if id_ not in d.keys():
        d[id_]={}
        d[id_]["MBE"]=np.sum(dif,axis=0)
        d[id_]["MSE"]=np.sum(np.power(dif,2),axis=0)
        d[id_]["MAE"]=np.sum(np.abs(dif),axis=0)
        d[id_]["MD"]=np.sum(dif_mag,axis=0)
        d[id_]["CD"]=np.sum(dif_cos,axis=0)
        d2[id_]=np.mean(np.power(dif,2),axis=(1,2,3))
    else:
        d[id_]["MBE"]+=np.sum(dif,axis=0)
        d[id_]["MSE"]+=np.sum(np.power(dif,2),axis=0)
        d[id_]["MAE"]+=np.sum(np.abs(dif),axis=0)
        d[id_]["MD"]+=np.sum(dif_mag,axis=0)
        d[id_]["CD"]+=np.sum(dif_cos,axis=0)
        d2[id_]=np.concatenate([d2[id_],np.mean(np.power(dif,2),axis=(1,2,3))])
    return d,d2


    
def get_test_error(out_ids,ids,HR_files,grid={},diri_out="/nfs/data/dahl0071/thesis/data/out",plot=0):
    #grid: dict storing WRF LR and HR grid info for baseline interpolators. If empty (default) no baselines calculated
    #idea to plot all on e.g. 4x1 subplot, storing SR results of all models in dict that gets updated for every HR file
    d={}#init
    d2={}
    N=0
    for j,file in enumerate(HR_files):
        d_results={} #init
        d_results["HR"]=np.load(file)
        HR_norm=np.linalg.norm(d_results["HR"],axis=-1)
        timestamp=file.split("/")[-1].split(".")[0]
        print("starting on "+timestamp)
        N+=np.shape(d_results["HR"])[0]
        for i, out_id in enumerate(out_ids):
            id_=ids[i]
            out_path='/'.join([diri_out, out_id])
            SR_file="/".join([out_path,timestamp+"_dataSR.npy"])   
            d_results[id_]=np.load(SR_file)
            d,d2=get_metrics(d,d2,d_results,HR_norm,id_)
#            dif=d_results["HR"]-d_results[id_]
#            SR_norm=np.linalg.norm(d_results[id_],axis=-1)
#            dif_mag=HR_norm-SR_norm
#            dif_cos=np.sum(d_results["HR"]*d_results[id_],axis=-1)/(HR_norm*SR_norm)
#            if id_ not in d.keys():
#                d[id_]={}
#                d[id_]["MBE"]=np.sum(dif,axis=0)
#                d[id_]["MSE"]=np.sum(np.power(dif,2),axis=0)
#                d[id_]["MAE"]=np.sum(np.abs(dif),axis=0)
#                d[id_]["MD"]=np.sum(dif_mag,axis=0)
#                d[id_]["CD"]=np.sum(dif_cos,axis=0)
#                d2[id_]=np.mean(np.power(dif,2),axis=(1,2,3))
#            else:
#                d[id_]["MBE"]+=np.sum(dif,axis=0)
#                d[id_]["MSE"]+=np.sum(np.power(dif,2),axis=0)
#                d[id_]["MAE"]+=np.sum(np.abs(dif),axis=0)
#                d[id_]["MD"]+=np.sum(dif_mag,axis=0)
#                d[id_]["CD"]+=np.sum(dif_cos,axis=0)
#                d2[id_]=np.concatenate([d2[id_],np.mean(np.power(dif,2),axis=(1,2,3))])
        # baseline interpolators, if grid dir not empty
        if bool(grid):
            print("Doing baselines")
            d_results["LR"]=downscale_image(d_results["HR"],10)
            
            #bicubic
            d_results["Bicubic"]=get_bicubic_interp_SR(grid,d_results["LR"])
            d,d2=get_metrics(d,d2,d_results,HR_norm,"Bicubic")
#            dif=d_results["HR"]-d_results["Bicubic"]
#            SR_norm=np.linalg.norm(d_results["Bicubic"],axis=-1)
#            dif_mag=HR_norm-SR_norm
#            dif_cos=np.sum(d_results["HR"]*d_results["Bicubic"],axis=-1)/(HR_norm*SR_norm)
#
#            if "Bicubic" not in d.keys():
#                d["Bicubic"]={}
#                d["Bicubic"]["MBE"]=np.sum(dif,axis=0)
#                d["Bicubic"]["MSE"]=np.sum(np.power(dif,2),axis=0)
#                d["Bicubic"]["MAE"]=np.sum(np.abs(dif),axis=0)
#                d["Bicubic"]["MD"]=np.sum(dif_mag,axis=0)
#                d["Bicubic"]["CD"]=np.sum(dif_cos,axis=0)
#                d2["Bicubic"]=np.mean(np.power(dif,2),axis=(1,2,3))
#            else:
#                d["Bicubic"]["MBE"]+=np.sum(dif,axis=0)
#                d["Bicubic"]["MSE"]+=np.sum(np.power(dif,2),axis=0)
#                d["Bicubic"]["MAE"]+=np.sum(np.abs(dif),axis=0)
#                d["Bicubic"]["MD"]+=np.sum(dif_mag,axis=0)
#                d["Bicubic"]["CD"]+=np.sum(dif_cos,axis=0)
#                d2["Bicubic"]=np.concatenate([d2["Bicubic"],np.mean(np.power(dif,2),axis=(1,2,3))])
            
            ### RBF
#             xobs=np.meshgrid(grid["we_lr"],grid["sn_lr"])
#             xgrid=np.meshgrid(grid["we"],grid["sn"])
#             xflat = np.reshape(xgrid,(2, -1)).T
#             xobs = np.reshape(xobs,(2, -1)).T
#             N_sn=len(grid["sn"])
#             N_we=len(grid["we"])
            
#             SR=get_RBF_interp_SR(LR_val,xobs,xflat,N_sn,N_we)
#             dif=SR-HR
#             if "RBF" not in d.keys():
#                 d["RBF"]=dif
#             else:
#                 d["RBF"]=np.concatenate([d["RBF"],dif],axis=0)

        if plot and j%100==0: #plot every 10th day
            print("plotting ...")
            image_path="/".join([out_path,"imgs",timestamp])
            plot_SR_HR_data(d_results,image_path)
            

        print(timestamp+" done")

    # get mean
    for key in d.keys():
        for metric in d[key]:
            d[key][metric]/=N
    print("Number of samples: {}".format(N))    
    return d,d2


baseline=0
dom="GB"
diri_in='/'.join(["/nfs/data/dahl0071/thesis/data/test",dom,"final/*"])
#diri_in='/'.join(["/nfs/data/dahl0071/thesis/data/test/2017"])

HR_files=glob(diri_in+"/*.npy")

#out_ids=["wind-20220524-135041","wind-20220527-191118","wind-20220528-031534"]
#ids=["CNN1 - no static", "CNN1 - HR static", "CNN2 - HR static"]

#out_ids=["wind-20220403-215439_20220404-104804","wind-20220519-113058","wind-20220520-215250","wind-20220419-143003_20220420-182503","wind-20220413-182848"]
#out_ids=["wind-20220430-180155","wind-20220429-211329"]
#ids=["GCNN1 - no static", "GCNN1 - HR and LR static", "GCNN1 - HR static","GCNN2 - no static","GCNN2 - HR and LR static"]
#ids=["GCNN3 - HR static","GCNN3 - HR and LR static"]


out_ids=["wind-20220528-031534","wind-20220602-122733","wind-20220603-122238"]
ids=["CNN2 - HR static","WGAN - no static","WGAN - HR static"]
if dom=="CE":
    ncfile="/nfs/data/dahl0071/thesis/data/ncfiles/P-CE-2017-2017-01-01.nc"
else:
    ncfile="/nfs/data/dahl0071/thesis/data/ncfiles/P-GB-2018-2018-12-17.nc"

if baseline:
    ds=xr.open_dataset(ncfile)
    if dom=="CE":
        ds=ds.isel(south_north=slice(2,412))
    else:
        ds=ds.isel(west_east=slice(3,393))
    LR=ds.WS.coarsen(south_north=10,west_east=10).mean()
    grid_d={}#init
    grid_d["sn_lr"]=LR.south_north.values
    grid_d["we_lr"]=LR.west_east.values
    grid_d["sn"]=ds.south_north.values
    grid_d["we"]=ds.west_east.values
    ds.close()
else:
    grid_d={}

d,d_dom=get_test_error(out_ids,ids,HR_files,grid=grid_d,plot=1)

#for id_ in d.keys():
#    dif=d[id_]
#    d[id_]={}
#    d[id_]["MSE"]=np.mean(np.power(dif,2),axis=0)
#    d[id_]["MBE"]=np.mean(dif,axis=0)
#    d[id_]["MAE"]=np.mean(np.abs(dif),axis=0)
res_file="/nfs/data/dahl0071/thesis/data/results/"+"__".join(out_ids)+"-"+dom+"-NodewiseStat.json"
with open(res_file, 'w') as fp:
    json.dump(d, fp,cls=NumpyEncoder)


res_file="/nfs/data/dahl0071/thesis/data/results/"+"__".join(out_ids)+"-"+dom+"-DomainStat.json"
with open(res_file, 'w') as fp:
    json.dump(d_dom, fp,cls=NumpyEncoder)



