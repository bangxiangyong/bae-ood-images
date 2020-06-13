from torchvision import datasets, transforms
from baetorch.baetorch.plotting import *
from baetorch.baetorch.util.seed import bae_set_seed
from baetorch.baetorch.util.misc import load_bae_model, save_csv_pd
from baetorch.baetorch.evaluation import calc_auroc, calc_auprc
import pandas as pd
import os

#set cuda if available
use_cuda = torch.cuda.is_available()

#set seed for reproduciliblity
bae_set_seed(2020)

#load fashion mnist
test_samples= 100
train_batch_size = 100

#load BAE for selected training set
# train_set_name = "SVHN"
train_set_name = "FashionMNIST"
# train_set_name = "CIFAR"
# train_set_name = "MNIST"

stochastic_samples = 100

trained_model_folder ="trained_models/"+train_set_name+"/"
all_model_name = os.listdir(trained_model_folder)
filter_model_name = ".p"
all_model_name = [model for model in all_model_name if filter_model_name in model]

#black list models
data_transform = transforms.Compose([transforms.ToTensor()])
if train_set_name == "CIFAR":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data-cifar', train=True, download=True,
                       transform=data_transform
                       ), batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data-cifar', train=False, download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data-svhn', split="test", download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=True)
elif train_set_name=="SVHN":
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data-svhn', split="train", download=True,
                       transform=data_transform
                       ), batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data-svhn', split="test", download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data-cifar', train=False, download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=True)
elif train_set_name=="FashionMNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data-fashion-mnist', train=True, download=True, transform=data_transform), batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data-fashion-mnist', train=False, download=True, transform=data_transform), batch_size=test_samples, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data-mnist', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=test_samples, shuffle=True)
elif train_set_name == "MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data-mnist', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data-mnist', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=test_samples, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data-fashion-mnist', train=False, download=True, transform=data_transform), batch_size=test_samples, shuffle=True)


#run evaluation test of model on ood data set
plot_output_reshape_size = (32, 32, 3) if (train_set_name =="SVHN" or train_set_name =="CIFAR") else (28,28)

whitelist_models = pd.read_csv("results/"+train_set_name+"_whitelist_models.csv")

plot_all = False

for row_id, row in whitelist_models.iterrows():
    if row["train_set_name"] == train_set_name:
        trained_model_name = row["model_metaname"]

        bae_model = load_bae_model(trained_model_name, folder="trained_models/"+train_set_name+"/")

        bae_model.set_cuda(use_cuda)
        if bae_model.model_type =="stochastic":
            bae_model.num_samples = stochastic_samples

        results_test = bae_model.predict(test_loader)
        results_ood = bae_model.predict(ood_loader)

        auroc_list, fpr80_list, metric_auroc = calc_auroc(results_test,results_ood, exclude_keys=[])
        auprc_list, metric_auprc = calc_auprc(results_test,results_ood, exclude_keys=[])

        #convert to dataframe
        auroc_pd = pd.DataFrame([{key:val for key,val in zip(metric_auroc,auroc_list)}])
        auprc_pd = pd.DataFrame([{key:val for key,val in zip(metric_auroc,auprc_list)}])
        fpr80_pd = pd.DataFrame([{key:val for key,val in zip(metric_auroc,fpr80_list)}])

        #add column for model name
        for result_pd,result_name in zip([auroc_pd,auprc_pd,fpr80_pd],["auroc","auprc","fpr80"]):
            result_pd["model"] = trained_model_name
            #rearrange column to set `model` as the first column
            cols = ["model"]+metric_auroc
            result_pd = result_pd[cols]
            #check for file exist
            save_csv_pd(result_pd,train_set_name=train_set_name,title=result_name)

        #plot samples
        plot_samples_img(results_test, reshape_size=(plot_output_reshape_size), savefile=bae_model.model_name+"_TEST.png")
        plot_samples_img(results_ood, reshape_size=(plot_output_reshape_size), savefile=bae_model.model_name+"_OOD.png")

        if not plot_all:
            plt.close('all')














