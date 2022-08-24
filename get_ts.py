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


def get_point_data(d,data,id,idx_x,idx_y,init=0):
    if init:
        for i in range(len(idx_x)):
            d[id][str(i)]=data[:,idx_x[i],idx_y[i],:]
    else:
        for i in range(len(idx_x)):
            d[id][str(i)]=np.concatenate([d[id][str(i)],data[:,idx_x[i],idx_y[i],:]],axis=0)
    return d


def get_ts(out_ids,ids,HR_files,idx_x,idx_y,diri_out="/nfs/data/dahl0071/thesis/data/out",plot=0):
    #grid: dict storing WRF LR and HR grid info for baseline interpolators. If empty (default) no baselines calculated
    #idea to plot all on e.g. 4x1 subplot, storing SR results of all models in dict that gets updated for every HR file
    d={}
    for key in ids:
        d[key]={}
    d["HR"]={}
    HR=np.load(HR_files[0])
    d=get_point_data(d,HR,"HR",idx_x,idx_y,init=1)
    timestamp=HR_files[0].split("/")[-1].split(".")[0]
    print("starting on "+timestamp)
    for i, out_id in enumerate(out_ids):
        id_=ids[i]
        out_path='/'.join([diri_out, out_id])
        SR_file="/".join([out_path,timestamp+"_dataSR.npy"])
        dat=np.load(SR_file)
        d=get_point_data(d,dat,id_,idx_x,idx_y,init=1)
    
    
    for j,file in enumerate(HR_files[1:]):
        HR=np.load(file)
        d=get_point_data(d,HR,"HR",idx_x,idx_y)
        timestamp=file.split("/")[-1].split(".")[0]
        print("starting on "+timestamp)
        for i, out_id in enumerate(out_ids):
            id_=ids[i]
            out_path='/'.join([diri_out, out_id])
            SR_file="/".join([out_path,timestamp+"_dataSR.npy"])   
            dat=np.load(SR_file)
            d=get_point_data(d,dat,id_,idx_x,idx_y)

        print(timestamp+" done")

    return d


dom="CE"
diri_in='/'.join(["/nfs/data/dahl0071/thesis/data/test",dom,"final/*"])
#diri_in='/'.join(["/nfs/data/dahl0071/thesis/data/test/2017"])

HR_files=glob(diri_in+"/*.npy")
print(len(HR_files))
#out_ids=["wind-20220524-135041","wind-20220527-191118","wind-20220528-031534"]
#ids=["CNN1 - no static", "CNN1 - HR static", "CNN2 - HR static"]

#out_ids=["wind-20220403-215439_20220404-104804","wind-20220519-113058","wind-20220520-215250","wind-20220419-143003_20220420-182503","wind-20220413-182848"]
#out_ids=["wind-20220430-180155","wind-20220429-211329"]
#ids=["GCNN1 - no static", "GCNN1 - HR and LR static", "GCNN1 - HR static","GCNN2 - no static","GCNN2 - HR and LR static"]
#ids=["GCNN3 - HR static","GCNN3 - HR and LR static"]


out_ids=["wind-20220524-135041","wind-20220527-191118","wind-20220528-031534","wind-20220602-122733","wind-20220603-122238"]
ids=["CNN1 - no static", "CNN1 - HR static","CNN2 - HR static","WGAN - no static","WGAN - HR static"]

d=get_ts(out_ids,ids,HR_files,[270,320,80,230,240],[200,120,190,185,100])

res_file="/nfs/data/dahl0071/thesis/data/ts/"+"_".join(out_ids)+"_"+dom+".json"
with open(res_file, 'w') as fp:
    json.dump(d, fp,cls=NumpyEncoder)


