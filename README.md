# Bayesian Autoencoders for Out-of-Distribution (OOD) Detection 
Code complementary to our paper "Bayesian Autoencoders: Analysing and Fixing the Bernoulli likelihood for Out-of-Distribution Detection"

## Models and likelihood
- Bayes by Backprop, Anchored ensembling, MC-Dropout
- Bernoulli, Continous Bernoulli, Gaussian

## Dataset pairs (in-distribution vs OOD)
- FashionMNIST vs MNIST
- MNIST vs FashionMNIST
- SVHN vs SVHN
- CIFAR10 vs CIFAR10

## Code description
- `train_models_main.py` for executing the training of the models. Creates a list of whitelisted models in `results`
- `test_models_images.py` tests the whitelisted models specified in `results/dataset_whitelist.csv` on in-distribution and OOD, and compute evaluation scores (area under curve of receiver operating characteristic -AUROC, area under the precision-recall curve (AUPRC), and false positive rate at 80% true positive rate)). These will be stored in a newly created `results` and `plots` folder
- `image_publication_pcc_ll.py` produces the figure for correlation between log-likelihood and proportion of zeros
- `image_publication_ism_hist.py` creates figure of histograms with various image similarity measures (binary cross entropy, mean-squared error, structural similarity measure and normalised mutual information)

#### Notes
- Remember to specify `train_set_name` variable to the desired in-distribution dataset. 
- Training options are commented out in `train_models_main.py`

## Figures

The following are samples from BAE, anchored ensembling (FashionMNIST vs MNIST)

### Image similarity histogram
![alt text](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/ism_hist.png)

### Maximum of Bernoulli log-likelihood varies with input, x
![alt text](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/max_ll.png)

### Checking for confounded by proportion of zeros in image
![alt text](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/pcc_ll.png)

### Reconstructed images
In-distribution (left panel), OOD (right panel)

![ID](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/id-outputs.png "in-distribution") ![OOD](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/ood-outputs.png "OOD")


