from PhIREGANs import *
from glob import glob
import argparse
import random


parser = argparse.ArgumentParser(description='Process cmd line input to training/test script.')
parser.add_argument('--isWGAN', action='store_true',default=False, help="Use WGAN ")
parser.add_argument('--isCNN', action='store_true',default=False, help="Only train generator (CNN)")
parser.add_argument('--use_mod_gen', action='store_true',default=False, help="Use modified generator")
parser.add_argument('--stoch_gen', action='store_true',default=False, help="Use input noise in generator")
parser.add_argument("--test_dir", nargs='?', type=str, help="Path to test data dir. Default is None, so if not provided then no testing after training")
parser.add_argument("--train_dir", nargs='?', type=str, help="Path to train data dir. MUST be provided")
parser.add_argument("--valid_dir", nargs='?', type=str, help="Path to validation data dir. MUST be provided")
parser.add_argument("--model_path", nargs='?', type=str, help="Path to trained model. Default is None, such that a new model is trained from scratch")
parser.add_argument("--HR_static_path", nargs='?', type=str, help="Path to HR static data. Default is None, such that the generator does not use HR static fields")
parser.add_argument("--LR_static_path", nargs='?', type=str, help="Path to LR static data. Default is None, such that the generator does not use LR static fields")
parser.add_argument("--epoch_shift", nargs='?', type=int, default=0, help="Epoch shift. Default is 0.")

parser.add_argument("--alpha_advers", nargs='?', type=float, default=1e-3, help="Adversarial loss factor")
parser.add_argument("--lr", nargs='?', type=float, default=1e-4, help="Learning rate used in optimizer during training")
parser.add_argument("--bs", nargs='?', type=int, default=64, help="Batchsize during training")
parser.add_argument("--ne", nargs='?', type=int, default=10, help="Number of epochs for training, default is 10")
parser.add_argument("--Ndisc", nargs='?', type=int, default=5, help="Number of WGAN disc training steps against generator updates, default is 5")
parser.add_argument("WGANlimit",nargs='?', type=float, default=2e3, help="Number of times the generator is updated at every iteration in the beg of train. def is 2000")

args=parser.parse_args()
print(args)

assert args.train_dir is not None, "Train dir must be provided"
if args.train_dir[-1] is not "/":
    args.train_dir+="/"

assert args.valid_dir is not None, "Validation dir must be provided"
if args.valid_dir[-1] is not "/":
    args.valid_dir+="/"


#data_path="/nfs/data/dahl0071/thesis/data/train/2018/"
train_data=glob(args.train_dir+"*.tfrecord")
assert len(train_data)>0, "No files found under provided train data dir"
print("found {} training files".format(len(train_data)))

valid_data=glob(args.valid_dir+"*.tfrecord")
assert len(valid_data)>0, "No files found under provided validation data dir"
print("found {} validation files".format(len(valid_data)))
#valid_data=random(valid_data,60) #randomly sample 60 files

if args.test_dir is not None:
    if args.test_dir[-1] is not "/":
        args.test_dir+="/"

    test_data=glob(args.test_dir+"*.tfrecord")
    if len(test_data)==0:
        print("No files found under provided test data dir, no testing")
        args.test_dir=None
    else:
        print("found {} test files".format(len(test_data)))
else:
    print("No test files provided")
data_type = 'wind'
#model_path = None
r = [2, 5]

#lr=1e-4 #1e-4
#bs=64
##for CV splits, many dirs inside CV dir. Search in CV folder for musig
if args.train_dir[-2]=="*":
    args.train_dir=args.train_dir[:-2]
try:
    mu_sig=np.load(args.train_dir+"mu_sig.npy")
except:
    print("mu_sig.npy not found under provided training data dir\ncalculating statistics from snapshots of training data")
    mu_sig=None    
if __name__ == '__main__':

    phiregans = PhIREGANs(data_type=data_type,N_epochs=args.ne,learning_rate=args.lr,save_every=5,mu_sig=mu_sig,use_mod_gen=args.use_mod_gen,epoch_shift=args.epoch_shift, is_stochastic=args.stoch_gen)
    #phiregans.set_static_fields("/nfs/data/dahl0071/thesis/data/static/CE/LR.npy","/nfs/data/dahl0071/thesis/data/static/CE/HR.npy")
    phiregans.set_static_fields(args.LR_static_path,args.HR_static_path)
    if args.isCNN:
        model_dir = phiregans.pretrain(r=r,
                                       data_path=train_data,
                                       valid_data_path=valid_data,
                                       model_path=args.model_path,
                                       batch_size=args.bs)
    else:
        #add condition using wgan or not (or remove is-wgan input arg) add alpha_advers and WGANlimit to args and pass to trainWGAN
        if args.isWGAN:
            model_dir = phiregans.trainWGAN(r=r,
                                            data_path=train_data,
                                            valid_data_path=valid_data,
                                            model_path=args.model_path,
                                            batch_size=args.bs,
                                            alpha_advers=args.alpha_advers,
					    WGANlimit=args.WGANlimit,
                                            Ndisc=args.Ndisc)
        else:
            assert args.model_path is not None, "Path to pretrained generator must be provided if not using WGAN"
            model_dir = phiregans.train(r=r,
                                        data_path=train_data,
                                        model_path=args.model_path,
                                        batch_size=args.bs,
                                        alpha_advers=args.alpha_advers)
    if args.test_dir is not None:
        phiregans.set_data_out_path('/'.join(['/nfs/data/dahl0071/thesis/data/out', phiregans.run_id]))
        i=0
        for file in test_data:
            if i%10==0:
                plot_bool=1
            else:
                plot_bool=0
            phiregans.test(r=r,
                           data_path=file,
                           model_path=model_dir,
                           batch_size=1,
                           plot_data=plot_bool)
            i+=1

    if mu_sig is None:
        print("storing mu_sig=\n",phiregans.mu_sig)
        np.save(args.train_dir+"mu_sig.npy",phiregans.mu_sig)

