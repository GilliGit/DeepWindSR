
from PhIREGANs import *
from glob import glob

data_path="/nfs/data/dahl0071/thesis/data/2018"
data_files=glob(data_path+"/*.tfrecord")
data_type = 'wind'
cnn_model_path = 'models/wind-20220306-032715/cnn/cnn'
gan_model_path = 'models/wind_lr-mr/trained_gan/gan'
r = [2, 5]

if __name__ == '__main__':

    phiregans = PhIREGANs(data_type=data_type,N_epochs=0)
    
    phiregans.model_name = gan_model_path[0:-4]

#    cnn_model_dir = phiregans.pretrain(r=r,
#                                       data_path=data_files,
#                                       model_path=cnn_model_path,
#                                       batch_size=20)

#    gan_model_dir = phiregans.train(r=r,
#                                    data_path=data_files,
#                                    model_path=gan_model_path,
#                                    batch_size=20)
    print("\n mu_sig=\n",phiregans.mu_sig)
    np.load(datapath+"/mu_sig.npy",phiregans.mu_sig)
