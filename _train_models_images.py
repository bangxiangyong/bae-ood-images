from baetorch.baetorch.models.base_autoencoder import *
from torchvision import datasets, transforms
from baetorch.baetorch.plotting import *
from baetorch.baetorch.test_suite import run_test_model
from baetorch.baetorch.lr_range_finder import run_auto_lr_range
from baetorch.baetorch.util.misc import save_bae_model

#model classes
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.bae_vi import BAE_VI
from baetorch.baetorch.models.bae_vi import VAE
from baetorch.baetorch.models.bae_mcdropout import BAE_MCDropout

from pprint import pprint

def train_model_images(train_set_name,likelihood_choice,model_type,latent_dim_multiplier,capacity_multiplier,decoder_sigma_choice,num_epoch_mu,num_epoch_sigma,latent_dim=10,conv_filters=[32,64,128],train_batch_size=100, use_cuda=False, weight_decay=0.1):
    #Update actual parameters based on selected choices
    if likelihood_choice == "bernoulli":
        likelihood ="bernoulli"
        homoscedestic_mode = "none"
        heteroscedestic = False
    elif likelihood_choice == "cbernoulli":
        likelihood ="cbernoulli"
        homoscedestic_mode = "none"
        heteroscedestic = False
    elif likelihood_choice == "gaussian_none":
        likelihood ="gaussian"
        homoscedestic_mode = "none"
        heteroscedestic = False
    elif likelihood_choice == "gaussian_homo":
        likelihood ="gaussian"
        homoscedestic_mode = "every"
        heteroscedestic = False
    elif likelihood_choice == "gaussian_hetero":
        likelihood ="gaussian"
        homoscedestic_mode = "none"
        heteroscedestic = True

    if train_set_name =="CIFAR":
        input_dim = 32
        input_channel =3
        conv_filters=[32,64,128]
        conv_kernel = [4,4,4]
        conv_stride = [2,1,2]

        plot_output_reshape_size=(32, 32, input_channel)

    elif train_set_name == "SVHN":

        input_dim = 32
        input_channel =3
        conv_filters=[32,64,128]
        conv_kernel = [4,4,4]
        conv_stride = [2,1,2]

        plot_output_reshape_size=(32, 32, input_channel)


    elif train_set_name =="FashionMNIST":
        input_dim = 28
        input_channel = 1
        conv_filters=[32,64]
        conv_kernel = [4,4]
        conv_stride = [2,1]
        plot_output_reshape_size=(28, 28)


    elif train_set_name =="MNIST":
        input_dim = 28
        input_channel = 1
        conv_filters=[32,64]
        conv_kernel = [4,4]
        conv_stride = [2,1]
        plot_output_reshape_size=(28, 28)


    conv_filters =[int(i*capacity_multiplier) for i in conv_filters]
    latent_dim = int(latent_dim*latent_dim_multiplier)

    #----LOAD DATA------
    test_samples= 1000
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
    elif train_set_name == "FashionMNIST":
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

    #----SPECIFY MODEL---------

    #model architecture
    conv_architecture=[input_channel]+conv_filters

    #specify encoder
    #with convolutional layers and hidden dense layer
    encoder = Encoder([ConvLayers(input_dim=input_dim,conv_architecture=conv_architecture, conv_kernel=conv_kernel,
                                  conv_stride=conv_stride, activation="leakyrelu", last_activation="sigmoid"),
               DenseLayers(architecture=[],output_size=latent_dim)])


    #specify decoder-mu and sigma
    decoder_mu = infer_decoder(encoder,last_activation="sigmoid") #symmetrical to encoder
    if decoder_sigma_choice == "infer":
        decoder_sig = infer_decoder(encoder, last_activation="none")
    elif decoder_sigma_choice == "dense":
        decoder_sig = DenseLayers(input_size=latent_dim,
                                  output_size=input_channel*input_dim*input_dim,
                                  architecture=[100,500]) #dense layers only

    #combine them into autoencoder
    if heteroscedestic == False:
        autoencoder = Autoencoder(encoder, decoder_mu)
    else:
        autoencoder = Autoencoder(encoder, decoder_mu, decoder_sig)


    #infer metadata name of model
    if weight_decay >= 1:
        weight_decay = int(weight_decay)
    model_metaname = model_type+"_"+train_set_name+"_L"+str(latent_dim)+"_"+likelihood_choice+"_"+"wd"+str(weight_decay).replace(".","")

    #instantiate BAE model
    if model_type=="vanilla":
        bae_model = BAE_Ensemble(model_name=model_metaname,
                                 autoencoder=autoencoder, use_cuda=use_cuda,
                                 anchored=True, weight_decay=weight_decay,
                                 num_samples=1, homoscedestic_mode=homoscedestic_mode,
                                 likelihood=likelihood)
    elif model_type =="ensemble":
        bae_model = BAE_Ensemble(model_name=model_metaname,
                                 autoencoder=autoencoder, use_cuda=use_cuda,
                                 anchored=True, weight_decay=weight_decay,
                                 num_samples=5, homoscedestic_mode=homoscedestic_mode,
                                 likelihood=likelihood)
    elif model_type =="mcdropout":
        bae_model = BAE_MCDropout(model_name=model_metaname, autoencoder=autoencoder,
                                  dropout_p=0.1, weight_decay=weight_decay, anchored=True,
                                      num_train_samples=5, num_samples=10, use_cuda=use_cuda, homoscedestic_mode=homoscedestic_mode, likelihood=likelihood)
    elif model_type =="vi":
        bae_model = BAE_VI(model_name=model_metaname, autoencoder=autoencoder,
                        num_train_samples=5,
                        num_samples=10, #during prediction only
                        use_cuda=use_cuda,
                        weight_decay=weight_decay, homoscedestic_mode=homoscedestic_mode, likelihood=likelihood)
    elif model_type =="vae":
        bae_model = VAE(model_name=model_metaname, autoencoder=autoencoder,
                        num_train_samples=5,
                        num_samples=10, #during prediction only
                        use_cuda=use_cuda, beta=weight_decay,
                        weight_decay=weight_decay, homoscedestic_mode=homoscedestic_mode, likelihood=likelihood)

    #save model meta data
    model_metadata = {  "model_metaname": model_metaname,
                        "train_set_name":train_set_name,
                         "model_type":model_type,
                         "num_epoch_mu":num_epoch_mu,
                         "latent_dim":latent_dim,
                         "capacity": str(conv_filters),
                         "likelihood":likelihood,
                         "weight_decay": weight_decay,
                         }
    bae_model.metadata = model_metadata

    print("TRAINING MODEL METADATA:")
    pprint(model_metadata)

    #------TRAINING------
    #train mu network
    try:
        run_auto_lr_range(train_loader, bae_model,run_full=False)
    except Exception as e:
        print(e)
        run_auto_lr_range(train_loader, bae_model,run_full=True)
    bae_model.fit(train_loader,num_epochs=num_epoch_mu)

    #train sigma network
    if bae_model.decoder_sigma_enabled:
        bae_model.scheduler_enabled = False #need to set scheduler off
        bae_model.learning_rate_sig = 1e-5 #set it to constant learning rate
        bae_model.fit(train_loader,num_epochs=num_epoch_sigma, mode="sigma", sigma_train="separate")

    #if model has reached nan in its loss
    if np.isnan(np.sum(bae_model.losses)):
        raise ValueError("NaN in training loss")

    #------PRE EVALUATION----
    #for each model, evaluate and plot:
    bae_models = [bae_model]
    id_data_test = test_loader
    ood_data_list = [ood_loader]
    ood_names = ["OOD"]

    #run evaluation test of model on ood data set
    run_test_model(bae_models=bae_models, id_data_test=test_loader,
                   ood_data_list=ood_data_list, id_data_name=train_set_name, ood_data_names=ood_names,
                   output_reshape_size=plot_output_reshape_size)

    #save model
    save_bae_model(bae_model, folder="trained_models/"+train_set_name+"/")

    return bae_model
