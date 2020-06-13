from baetorch.baetorch.models.base_autoencoder import *
from torchvision import datasets, transforms
from baetorch.baetorch.plotting import *
from baetorch.baetorch.util.seed import bae_set_seed
from baetorch.baetorch.util.misc import load_bae_model
from baetorch.baetorch.util.distributions import CB_Distribution

import os
from scipy import stats


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

shuffle =False

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


#set names of chosen models here for each likelihood:
model_likelihood = {"FashionMNIST":{"bernoulli":"ensemble_FashionMNIST_L20_bernoulli_wd1",
                    "cbernoulli":"ensemble_FashionMNIST_L20_cbernoulli_wd1",
                    "gaussian":"ensemble_FashionMNIST_L20_gaussian_none_wd1"
                     }}

results_key = {"bernoulli":["bce_mean","bce_var","input"],
               "cbernoulli":["cbce_mean","cbce_var"],
               "gaussian":["se_mean","se_var"]}
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
#=====min-max bernoulli=====
cb_dist = CB_Distribution()
dpi =500
const_range = np.linspace(0.0000000001, 0.999999999, 1000)
bernoulli_range_max = [-log_bernoulli_loss_np(const.reshape(1, 1), const.reshape(1, 1)).item() for const in const_range]
cbernoulli_range_max = [-cb_dist.log_cbernoulli_loss_np(torch.Tensor(const.reshape(1, 1)), torch.Tensor(const.reshape(1, 1))).item() for const in const_range]
gaussian_range_max = [0 for const in const_range]

def plot_maximum_LL(const_range, range_max, dpi=500, figsize_scale=2.5):
    figsize_scale = 2.5
    figsize = (int(1.5*figsize_scale),int(1*figsize_scale))
    plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(const_range, range_max)
    font_size_label = "small"
    plt.xlabel("x",fontsize=font_size_label)
    plt.ylabel("Maximum Log-likelihood",fontsize=font_size_label)

    plt.tight_layout()

# plot single graph
# plot_maximum_LL(const_range,bernoulli_range_max)
# plot_maximum_LL(const_range,cbernoulli_range_max)

#plot 3 graphs side-by-side
plt.figure()
figsize_scale = 2.5
figsize = (int(4*figsize_scale),int(1*figsize_scale))
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=figsize,dpi=dpi)
ax1.plot(const_range, bernoulli_range_max)
ax2.plot(const_range, cbernoulli_range_max)
ax3.plot(const_range, gaussian_range_max)

ax3.set_yticks((1.0,0,-1.0))
ax3.set_yticklabels(("",r"$-\frac{1}{2}\log{\sigma_i^2}$",""), fontsize=5)
font_size_label = "small"
ax2.set_xlabel("x",fontsize=font_size_label)
ax1.set_ylabel("Maximum Log-likelihood",fontsize=font_size_label)

font_size_title = "x-small"
ax1.set_title("a) Bernoulli", fontsize=font_size_title)
ax2.set_title("b) C-Bernoulli", fontsize=font_size_title)
ax3.set_title("c) Gaussian", fontsize=font_size_title)

plt.tight_layout()
plt.subplots_adjust(top=0.846,
bottom=0.244,
left=0.131,
right=0.985,
hspace=0.2,
wspace=0.505)

##===correlation graphs====

#figure settings parameters
dpi = 300
figsize_scale = 3.5
pcc_graph_figsize = (int(1.5*figsize_scale),int(1*figsize_scale))
point_size = 3
edgecolors= None
point_alpha = 0.75
font_size_pcc = 8
pcc_text_x = 0.5
pcc_text_y = 0.93
constrained_layout=False

legend_fontsize="xx-small"
font_size_title = "small"

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='xx-small')
plt.rc('ytick', labelsize='xx-small')


#calculations
def calc_proportion_zeros(image,threshold=0.):
    image = image.reshape(-1)
    total_pixels = image.shape[0]
    proportion_zeros = np.argwhere(image<=threshold).shape[0]/total_pixels
    return proportion_zeros

proportion_zeros_test = np.apply_along_axis(calc_proportion_zeros,1, flatten_np(results_test["input"]))
proportion_zeros_ood = np.apply_along_axis(calc_proportion_zeros,1, flatten_np(results_ood["input"]))

#bce_mean and se_mean
bce_mean_test = flatten_np(results_test["bce_mean"]).mean(1)
bce_mean_ood = flatten_np(results_ood["bce_mean"]).mean(1)
cbce_mean_test = flatten_np(results_test["cbce_mean"]).mean(1)
cbce_mean_ood = flatten_np(results_ood["cbce_mean"]).mean(1)
se_mean_test = flatten_np(results_test["se_mean"]).mean(1)
se_mean_ood = flatten_np(results_ood["se_mean"]).mean(1)
bce_var_test = flatten_np(results_test["bce_var"]).mean(1)
bce_var_ood = flatten_np(results_ood["bce_var"]).mean(1)
cbce_var_test = flatten_np(results_test["cbce_var"]).mean(1)
cbce_var_ood = flatten_np(results_ood["cbce_var"]).mean(1)
se_var_test = flatten_np(results_test["se_var"]).mean(1)
se_var_ood = flatten_np(results_ood["se_var"]).mean(1)

pcc_bce_zeros = stats.pearsonr(np.append(proportion_zeros_test,proportion_zeros_ood),
                         np.append(-bce_mean_test,-bce_mean_ood))[0]
pcc_cbce_zeros = stats.pearsonr(np.append(proportion_zeros_test,proportion_zeros_ood),
                         np.append(-cbce_mean_test,-cbce_mean_ood))[0]
pcc_mse_zeros = stats.pearsonr(np.append(proportion_zeros_test,proportion_zeros_ood),
                         np.append(-se_mean_test,-se_mean_ood))[0]
pcc_mse_var = stats.pearsonr(np.append(proportion_zeros_test,proportion_zeros_ood),
                         np.append(se_var_test,se_var_ood))[0]
pcc_bce_var = stats.pearsonr(np.append(proportion_zeros_test,proportion_zeros_ood),
                         np.append(bce_var_test,bce_var_ood))[0]
pcc_cbce_var = stats.pearsonr(np.append(proportion_zeros_test,proportion_zeros_ood),
                         np.append(cbce_var_test,cbce_var_ood))[0]

fig, ((ax11,ax12,ax13), (ax21,ax22,ax23)) = plt.subplots(2,3, figsize=pcc_graph_figsize,dpi=dpi, constrained_layout=constrained_layout)

test_markers = ax11.scatter(proportion_zeros_test,-bce_mean_test, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ood_markers = ax11.scatter(proportion_zeros_ood,-bce_mean_ood, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax12.scatter(proportion_zeros_test,-cbce_mean_test, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax12.scatter(proportion_zeros_ood,-cbce_mean_ood, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax13.scatter(proportion_zeros_test,-se_mean_test, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax13.scatter(proportion_zeros_ood,-se_mean_ood, s=point_size, alpha=point_alpha,edgecolors=edgecolors)

ax21.scatter(proportion_zeros_test, bce_var_test, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax21.scatter(proportion_zeros_ood, bce_var_ood, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax22.scatter(proportion_zeros_test, cbce_var_test, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax22.scatter(proportion_zeros_ood, cbce_var_ood, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax23.scatter(proportion_zeros_test,se_var_test, s=point_size, alpha=point_alpha,edgecolors=edgecolors)
ax23.scatter(proportion_zeros_ood,se_var_ood, s=point_size, alpha=point_alpha,edgecolors=edgecolors)

ax13.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax23.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

for ax in (ax11,ax12,ax13,ax21,ax22,ax23):
    ax.set_xlim(0.0,1.0)
    ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])


#set legends
test_label = "FashionMNIST"
ood_label = "MNIST"

def set_legend_marksersize(lgnd, fixed_size=30):
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [fixed_size]

#set x-y and title labels
ax11.set_ylabel(r"$E_{\theta}(LL)$", fontsize=font_size_title)
ax21.set_ylabel(r"$Var_{\theta}(LL)$", fontsize=font_size_title)
ax22.set_xlabel("Proportion of zeros in an image", fontsize=font_size_title)

ax11.set_title("Bernoulli", fontsize=font_size_title)
ax12.set_title("C-Bernoulli", fontsize=font_size_title)
ax13.set_title("Gaussian", fontsize=font_size_title)

def str_deci(number,num_deci=2):
    return ("{0:."+str(num_deci)+"f}").format(round(number,num_deci))

ax11.text(pcc_text_x, pcc_text_y,'PCC='+str_deci(pcc_bce_zeros,2), ha='center', va='center',
          transform=ax11.transAxes,fontsize=font_size_pcc)
ax12.text(pcc_text_x, pcc_text_y,'PCC='+str_deci(pcc_cbce_zeros,2), ha='center', va='center',
          transform=ax12.transAxes,fontsize=font_size_pcc)
ax13.text(pcc_text_x, pcc_text_y,'PCC='+str_deci(pcc_mse_zeros,2), ha='center', va='center',
          transform=ax13.transAxes,fontsize=font_size_pcc)
ax21.text(pcc_text_x, pcc_text_y,'PCC='+str_deci(pcc_bce_var,2), ha='center', va='center',
          transform=ax21.transAxes,fontsize=font_size_pcc)
ax22.text(pcc_text_x, pcc_text_y,'PCC='+str_deci(pcc_cbce_var,2), ha='center', va='center',
          transform=ax22.transAxes,fontsize=font_size_pcc)
ax23.text(pcc_text_x, pcc_text_y,'PCC='+str_deci(pcc_mse_var,2), ha='center', va='center',
          transform=ax23.transAxes,fontsize=font_size_pcc)

set_legend_marksersize(fig.legend([test_markers, ood_markers],
                                  labels=[test_label,ood_label],
                                  loc="upper left",
                                  borderaxespad=0.1,
                                  fontsize=legend_fontsize))

plt.tight_layout()
plt.subplots_adjust(top=0.814,
bottom=0.124,
left=0.102,
right=0.975,
hspace=0.377,
wspace=0.365)

