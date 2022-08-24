import ast
from PhIREGANs import *
from glob import glob
from time import strftime
import argparse
import os

parser = argparse.ArgumentParser(description='Process cmd line input to test script.')
parser.add_argument("--log", nargs='?', type=str, help="Path to log file, relative to thesis work dir. Must be provided.")
parser.add_argument("--test_dir", nargs='?', type=str, help="Path to test data dir. Must be provided.")
parser.add_argument("--model_name", nargs='?', type=str, help="Model name. Must be provided.")
parser.add_argument("--dom", nargs='?', type=str, default="CE",help="Domain name. Must be either GB or CE (latter is default)")

args=parser.parse_args()
print(args)

assert args.log is not None, "log file must be provided"
assert args.test_dir is not None, "test data dir must be provided"
assert args.model_name is not None, "model name must be provided"
assert args.dom in ["CE","GB"], "domain should be CE or GB"

with open("/".join(['/gss/work/dahl0071/thesis/logs',args.log])) as f:
    namespace = f.readline()
d={}
info=namespace[10:-2].split(", ")
for param in info:
    split=param.split("=")
    d[split[0]]=ast.literal_eval(split[1])
print("Following namespace params found \n {}".format(d))

if args.test_dir[-1] is not "/":
    args.test_dir+="/"

test_data=glob(args.test_dir+"*.tfrecord")
assert len(test_data)>0, "No files found under provided test data dir, no testing"
print("found {} test files".format(len(test_data)))

#test_data_files = glob("/nfs/data/dahl0071/thesis/data/test/2017/*.tfrecord")
#print("Files for testing:\n")
#print(test_data_files)

data_type = 'wind'
#model_name="wind-20220419-143003"#"wind-20220411-021151"#"wind-20220404-011634"#"wind-20220403-215439" #"wind-20220404-011634"

model_path = "/".join(['models',args.model_name])#,"cnn",'cnn'])
if d["isCNN"]:
    model_path = "/".join([model_path,"cnn","cnn"])
else:
    model_path ="/".join([model_path,"gan",'gan'])

r = [2, 5]

if d["train_dir"][-2]=="*":
    d["train_dir"]=d["train_dir"][:-2]
try:
    mu_sig=np.load(d["train_dir"]+"mu_sig.npy")
except:
    assert 1==2, "mu_sig.npy not found under provided training data dir"
    
if args.dom=="GB":
    if d["LR_static_path"] is not None:
        d["LR_static_path"]=d["LR_static_path"].replace("CE","GB")
    if d["HR_static_path"] is not None:
        d["HR_static_path"]=d["HR_static_path"].replace("CE","GB")

#mu_sig=np.load("/nfs/data/dahl0071/thesis/data/train/2018/mu_sig.npy")

if __name__ == '__main__':

    phiregans = PhIREGANs(data_type=data_type,mu_sig=mu_sig,use_mod_gen=d["use_mod_gen"],is_stochastic=d["stoch_gen"])
    phiregans.set_static_fields(d["LR_static_path"],d["HR_static_path"])
    
    phiregans.set_data_out_path('/'.join(['/nfs/data/dahl0071/thesis/data/out',args.model_name])) #"_".join([model_name,strftime('%Y%m%d-%H%M%S')])]))
#    phiregans.set_static_fields("/nfs/data/dahl0071/thesis/data/static/CE/LR.npy","/nfs/data/dahl0071/thesis/data/static/CE/HR.npy")
    print(phiregans.data_out_path) 
    if d["stoch_gen"]:
        test_data=test_data
        print(test_data)
        for i in range(100):
            phiregans.test(r=r,
                           data_path=test_data,
                           model_path=model_path,
                           batch_size=1,
                           fn="stoch_{0}_{1}".format(args.dom,i))
    else:
        i=0
        for file in test_data:
            if i<10:
                plot_bool=1
            else:
                plot_bool=0
            phiregans.test(r=r,
                           data_path=file,
                           model_path=model_path,
                           batch_size=1,
                           plot_data=plot_bool)
            i+=1

