from baetorch.baetorch.models.base_autoencoder import *
from torchvision import datasets, transforms
from baetorch.baetorch.plotting import *
from baetorch.baetorch.util.seed import bae_set_seed
from baetorch.baetorch.util.misc import load_bae_model
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

shuffle =True

#black list models

data_transform = transforms.Compose([transforms.ToTensor()])
if train_set_name == "CIFAR":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data-cifar', train=True, download=True,
                       transform=data_transform
                       ), batch_size=train_batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data-cifar', train=False, download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=shuffle)
    ood_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data-svhn', split="test", download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=shuffle)
elif train_set_name=="SVHN":
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data-svhn', split="train", download=True,
                       transform=data_transform
                       ), batch_size=train_batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data-svhn', split="test", download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=shuffle)
    ood_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data-cifar', train=False, download=True,
                       transform=data_transform
                       ), batch_size=test_samples, shuffle=shuffle)
elif train_set_name=="FashionMNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data-fashion-mnist', train=True, download=True,
                              transform=data_transform), batch_size=train_batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data-fashion-mnist', train=False, download=True,
                              transform=data_transform), batch_size=test_samples, shuffle=shuffle)
    ood_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data-mnist', train=False, download=True,
                       transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=test_samples, shuffle=shuffle)
elif train_set_name == "MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data-mnist', train=True, download=True,
                       transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=train_batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data-mnist', train=False, download=True,
                       transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=test_samples, shuffle=shuffle)
    ood_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data-fashion-mnist', train=False,
                              download=True, transform=data_transform), batch_size=test_samples, shuffle=shuffle)


#run evaluation test of model on ood data set
plot_output_reshape_size = (32, 32, 3) if (train_set_name =="SVHN" or train_set_name =="CIFAR") else (28,28)


#for each model, evaluate and plot:
model_likelihood = {"FashionMNIST":{"bernoulli":"ensemble_FashionMNIST_L20_bernoulli_wd1"}}
results_key = {"bernoulli":["mu","input","epistemic"]}
results_test = {}
results_ood = {}

for likelihood, trained_model_name in model_likelihood[train_set_name].items():
    bae_model = load_bae_model(trained_model_name, folder="trained_models/"+train_set_name+"/")

    bae_model.set_cuda(use_cuda)
    if bae_model.model_type =="stochastic":
        bae_model.num_samples = stochastic_samples

    temp_results_test = bae_model.predict(test_loader)
    temp_results_ood = bae_model.predict(ood_loader)

    results_test.update({key:temp_results_test[key] for key in results_key[likelihood]})
    results_ood.update({key:temp_results_ood[key] for key in results_key[likelihood]})

plot_samples_img(results_test, reshape_size=(plot_output_reshape_size), savefile=bae_model.model_name+"_TEST.png")
plot_samples_img(results_ood, reshape_size=(plot_output_reshape_size), savefile=bae_model.model_name+"_OOD.png")



def log_bernoulli_loss(y_pred, y_true, reduction="none"):
    bce = -(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    bce = np.nan_to_num(bce, nan=0,posinf=100,neginf=-100)
    if reduction == "sum":
        return np.sum(bce)
    elif reduction == "mean":
        return np.mean(bce)
    else: #none
        return bce

def log_bernoulli_loss_np(y_pred, y_true):
    y_pred = flatten_np(y_pred)
    y_true = flatten_np(y_true)
    bce = -(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    bce = np.nan_to_num(bce, nan=0,posinf=100,neginf=-100)
    return bce

def log_gaussian_loss(y_pred_mu, y_true):
    y_pred_mu = flatten_np(y_pred_mu)
    y_true = flatten_np(y_true)
    mse = (y_pred_mu-y_true)**2
    return mse

#PUBLICATION PLOTS
#figure settings parameters
#set legends
test_label = "FashionMNIST"
ood_label = "MNIST"

def set_legend_marksersize(lgnd, fixed_size=30):
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [fixed_size]


#=====HISTOGRAM OF IMAGE SIMILARITY MEASURES=======
from sklearn.metrics.cluster import normalized_mutual_info_score, mutual_info_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from scipy.stats import gaussian_kde

def mse(x, y,ax=-1):
    return np.mean((x - y)**2, axis=ax)

def calc_image_similarity(results_ood,results_test,method_id="mse"):
    def convert_image_int(image):
        return (image.flatten()*255).astype(int)

    ism_ood = []
    ism_test = []
    method_map = {"nmi":normalized_mutual_info_score,
                  "mi":mutual_info_score,
                  "ssim":ssim,
                  "bce":log_bernoulli_loss,
                  "psnr": peak_signal_noise_ratio,
                  "mse": mse
                  }
    method = method_map[method_id]

    for i in range(results_ood["mu"].shape[0]):
        if method_id == "nmi" or method_id == "mi":
            ism_score = method(convert_image_int(results_ood["mu"][i]), convert_image_int(results_ood["input"][i]))
        elif method_id == "bce" or method_id == "mse":
            ism_score = np.mean(method(results_ood["mu"][i].flatten(), results_ood["input"][i].flatten()))*-1
        else:
            ism_score = method(results_ood["mu"][i].flatten(), results_ood["input"][i].flatten())
        ism_ood.append(ism_score)

    for i in range(results_test["mu"].shape[0]):
        if method_id == "nmi" or method_id == "mi":
            ism_score = method(convert_image_int(results_test["mu"][i]), convert_image_int(results_test["input"][i]))
        elif method_id == "bce" or method_id == "mse":
            ism_score = np.mean(method(results_test["mu"][i].flatten(), results_test["input"][i].flatten()))*-1
        else:
            ism_score = method(results_test["mu"][i].flatten(), results_test["input"][i].flatten())
        ism_test.append(ism_score)
    return ism_test, ism_ood

#calculation
bce_test, bce_ood = calc_image_similarity(results_ood,results_test,method_id="bce")
mse_test, mse_ood = calc_image_similarity(results_ood,results_test,method_id="mse")
nmi_test, nmi_ood = calc_image_similarity(results_ood,results_test,method_id="nmi")
ssim_test, ssim_ood = calc_image_similarity(results_ood, results_test, method_id="ssim")

#actual plotting
dpi = 300
figsize_scale = 4
pcc_graph_figsize = (int(1.25*figsize_scale),int(1*figsize_scale))
ism_hist_alpha = 0.8

legend_fontsize=6
font_size_title = "small"

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

test_label = "FashionMNIST"
ood_label = "MNIST"

fig, ((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2, figsize=pcc_graph_figsize,dpi=dpi)

def ax_kde(ax, data):
    pd.DataFrame(data).plot(kind='density', ax=ax)


for ax_, ism_test,ism_ood in zip([ax11,ax12,ax21,ax22],
                                 [bce_test, mse_test, ssim_test, nmi_test],
                                 [bce_ood, mse_ood, ssim_ood, nmi_ood]):
    ax_kde(ax_, ism_test)
    ax_kde(ax_, ism_ood)
    ax_.legend([test_label,ood_label],fontsize=legend_fontsize)

ax11.set_ylabel("Histogram", fontsize=font_size_title)
ax21.set_ylabel("Histogram", fontsize=font_size_title)
ax22.set_ylabel("")
ax12.set_ylabel("")
ax11.set_xlabel("-BCE", fontsize=font_size_title)
ax12.set_xlabel("-MSE", fontsize=font_size_title)
ax21.set_xlabel("SSIM", fontsize=font_size_title)
ax22.set_xlabel("NMI", fontsize=font_size_title)
# plt.suptitle("c) Histograms")
plt.tight_layout()
plt.subplots_adjust(top=0.963,
bottom=0.127,
left=0.095,
right=0.966,
hspace=0.525,
wspace=0.186)
