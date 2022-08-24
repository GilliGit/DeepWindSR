import numpy as np
from glob import glob

def get_mu_sig(p0,dom,pout=None):
    '''
        Compute mean (mu) and standard deviation (sigma) for each data channel
        inputs:
            data_path - (string) path to the tfrecord for the training data
            batch_size - number of samples to grab each interation

        outputs:
            sets self.mu_sig
    '''
    #### change to use LR data aswell if using ERA5..
    files=glob(p0+"/P-"+dom+"*.npy")
    print("found {} files".format(len(files)))
    N, mu, mu3, sigma = 0, 0, 0, 0
    for file in files:
        data_HR=np.load(file)
        data_HR=np.linalg.norm(data_HR,axis=-1)
        N_batch, h, w = data_HR.shape
        N_new = N + N_batch

        mu_batch = np.mean(data_HR, axis=0)
        sigma_batch = np.var(data_HR, axis=0)
        mu3_batch = np.mean(data_HR**3, axis=0)

        sigma = (N/N_new)*sigma + (N_batch/N_new)*sigma_batch + (N*N_batch/N_new**2)*(mu - mu_batch)**2
        mu = (N/N_new)*mu + (N_batch/N_new)*mu_batch
        mu3 = (N/N_new)*mu3 + (N_batch/N_new)*mu3_batch

        N = N_new
    if pout is None:
        pout=p0
    np.save(pout+"/meanWS_{}.npy".format(dom),mu)
    np.save(pout+"/varWS_{}.npy".format(dom),sigma)
    np.save(pout+"/meanWScube_{}.npy".format(dom),mu3)

models=['wind-20220524-135041',"wind-20220527-191118",'wind-20220528-031534',"wind-20220602-122733","wind-20220603-122238"]

for model in models:
    print("starting on {}".format(model))
    get_mu_sig("/".join(["/nfs/data/dahl0071/thesis/data","out",model]),"CE")
    get_mu_sig("/".join(["/nfs/data/dahl0071/thesis/data","out",model]),"GB")

get_mu_sig("/nfs/data/dahl0071/thesis/data/test/CE/final/*","CE","/nfs/data/dahl0071/thesis/data/test/clima")
get_mu_sig("/nfs/data/dahl0071/thesis/data/test/GB/final/*","GB","/nfs/data/dahl0071/thesis/data/test/clima")
