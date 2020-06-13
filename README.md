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

## Figures

The following are samples from BAE, anchored ensembling (FashionMNIST vs MNIST)

### Image similarity histogram
![alt text](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/ism_hist.png)

### Maximum of Bernoulli log-likelihood varies with input, x
![alt text](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/max_ll.png)

### Checking for confounded by proportion of zeros in image
![alt text](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/pcc_ll.png)

### Reconstructed images

![ID](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/id-outputs.png "in-distribution") ![OOD](https://github.com/bangxiangyong/bae-ood-images/blob/master/figures/ood-outputs.png "OOD")


