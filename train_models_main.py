from baetorch.baetorch.plotting import *
from baetorch.baetorch.util.seed import bae_set_seed
from baetorch.baetorch.util.misc import save_csv_pd
from _train_models_images import train_model_images
import pandas as pd


#set seed for reproduciliblity
bae_set_seed(2020)
use_cuda = torch.cuda.is_available()

print("USE CUDA:"+str(use_cuda))

#----STATIC GLOBAL PARAMETERS---
#available choices -- static
decoder_sigma_choices=["infer","dense"]
latent_dim_multiplier_choices = [0.2,0.5,1,2,5]
capacity_multiplier_choices = [0.2,0.5,1,2,5]



#set likelihood and model hyperparameters
latent_dim_multiplier = latent_dim_multiplier_choices[2] #for easy scaling of latent dims & capacity
capacity_multiplier = capacity_multiplier_choices[2]
decoder_sigma_choice = decoder_sigma_choices[1] #aleatoric uncertainty, unused in this experiment runs
num_epoch_sigma = 0 #aleatoric uncertainty, unused

#Change here:
model_types_choices= ["vanilla","ensemble","mcdropout","vi","vae"]
likelihood_choices = ["bernoulli","cbernoulli","gaussian_none"]
train_set_choices = ["FashionMNIST","MNIST","SVHN","CIFAR"]
num_epoch_mu = 1 #number of training epochs
train_batch_size = 100
latent_dim =20

train_set_choices = ["FashionMNIST"] #["FashionMNIST","MNIST","SVHN","CIFAR"]
model_types_choices= ["ensemble"] #["vanilla","ensemble","mcdropout","vi","vae"]
likelihood_choices = ["bernoulli"] #["bernoulli","cbernoulli","gaussian_none"]
weight_decay_list = [1] #[10.,2.,1.,0.1,0.01,0.001]

#see STATIC PARAMETERS for the available choice

plot_all = False

for train_set_name in train_set_choices:
    #Select BAE models
    create_dir("trained_models/")
    create_dir("trained_models/"+train_set_name)
    for model_type in ["ensemble"]:
        #Select bernoulli vs gaussian likelihood
        for likelihood_choice in likelihood_choices:
            for weight_decay in weight_decay_list:
                try:

                    bae_model = train_model_images(train_set_name,likelihood_choice,model_type,latent_dim_multiplier,capacity_multiplier,decoder_sigma_choice,num_epoch_mu,num_epoch_sigma,use_cuda=use_cuda, latent_dim=latent_dim, weight_decay=weight_decay)
                    #save whitelisted model metadata
                    save_csv_pd(pd.DataFrame([bae_model.metadata]),train_set_name=train_set_name,title="whitelist_models", folder="results")

                except Exception as e:
                    print(e)
                    print("Error in training and testing model")

                if not plot_all:
                    plt.close('all')


