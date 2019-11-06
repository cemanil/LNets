# LNets
Implementation and evaluation of Lipschitz neural networks (LNets). Paper link: https://arxiv.org/abs/1811.05381 

# Installation
* Create a new conda environment and activate it:
    ```
    conda create -n lnets python=3.5
    conda activate lnets
    ```
    
* Install PyTorch, following instructions in `https://pytorch.org`. 

* Install torchnet by:
    ```
    pip install git+https://github.com/pytorch/tnt.git@master
    ```

* Navigate to the root of the project. Install the package, along with requirements:
    ```
    python setup.py install
    ```
    
* Add project root to PYTHONPATH. One way to do this: 
    ```
    export PYTHONPATH="${PYTHONPATH}:`pwd`"
    ``` 
    
**Note on PyTorch version**: All the experiments were performed using PyTorch version 0.4.1, although the code is expected
to run using Pytorch 1.0. 

# Models
Code that implements the core ideas presented in the paper are shown below. 

```
lnets
├── models
│   └── acivations
│       └── group_sort.py                   "GroupSort activation. " 
│       └── maxout.py                       "MaxOut and MaxMin activations. "
│   └── layers
│       └── conv
│           └── bjorck_conv2d.py            "Conv layer with Bjork-orthonormalized filters. "
│           └── l_inf_projected_conv2d.py   "Conv layer with L-infinity projected filters. "
│       └── dense
│           └── bjorck_linear.py            "Dense layer with Bjorck-orthonormalized weights. "
│           └── l_inf_projected.py          "Dense layer with l-infinity projected weights. "
│           └── parseval_l2_linear.py       "Dense layer with Parseval regularization. "
│           └── spectral_normal.py          "Dense layer with spectral normalization. "
│   └── regularization
│       └── __init__.py
│       └── spec_jac.py                     "Penalizes the jacobian norm. Description in Appendix of paper. "
│   └── utils
│       └── __init__.py
│       └── conversion.py                   "Converts a Bjorck layer to a regular one for fast test time inference. "
│   └── __init__.py                         "Specification of models for a variety of tasks. "
```

## Configuring Experiments
We strived to put as many variables as we could in a single configuration (json) file for each experiment. 
Sample configuration files exist under:
* `lnets/tasks/adversarial/configs`: for adversarial robustness experiments.
* `lnets/tasks/classification/configs`: for classification experimnts.
* `lnets/tasks/dualnets/configs`: for Wasserstein distance estimation experiments. 
* `lnets/tasks/gan/configs`: for training GANs. 

We now describe the key moving parts in these configs and how to change them.  

### Model Configuration
`model.name`: (string) Chooses the overall architecture and the task/training objective. `lnets/models/__init__.py` contains 
 the commonly used model names. Two examples are:
* "dual_fc": Train a fully connected model, under the dual Wasserstein objective. 
* "classify_fc": Train a fully connected classifier. 

`model.activation`: (string) Activation used throughout the network. One of "maxmin", "group_sort", "maxout", "relu", "tahn", 
"sigmoid" or "identity" (i.e. no activation). 

`model.linear.type`: (string) Chooses which linear layer type is going to be used. If the model is fully connected, the available
options are:
* "standard": The usual linear layer. 
* "bjorck": Bjorck orthonormalized - all singular values equal to one. 
* "l_inf_projected": Weight matrices are projected to the L-infinity ball. 
* "spectral_normal": Use spectral normalization - largest singular value set to 1. 
* "parseval_l2": Parseval regularized linear transformation.

If the architecture is fully convolutional, the available options are: 
* "standard_conv2d": the standard convolutional layer,
* "bjorck_conv2d": Convolutional layers in which the filters are Bjorck orthonormalized,
* "l_inf_projected_conv2d": Convolutional layers in which the filters are projected to the L-infinity ball. 

`model.layers`: (list) Contains how many neurons (or convolutional filters) there should be in each layer. 

`model.groupings`: (list) This field is used for activations that perform operations on groups of neurons. Used for GroupSort, 
MaxMin and MaxOut. Is a list specifying the grouping sizes for each layer. For example, setting to \[2, 3\] means the 
activation should act on groups of 2 and 3 in the first and second layers, respectively.

`l_constant`: (integer) Scales the output of each layer by a certain amount such that the network output is scaled by 
l_constant. Used to build K-Lipschitz networks out of 1-Lipschitz building blocks. 

`per_update_proj` and `per_epoch_proj`: Some algorithms (such as Parseval networks) involve projecting the weights of 
networks to a certain manifold after each training update. These fields let the user flexibly choose how often and with
which projection algorithm the weights should be projected. The supported projection algorithms are:
* "l_2": project to L2 ball. 
* "l_inf_projected": project to the L-infinity ball.
 
 By default, after-update and after-epoch updates are set to false. 
 
### Running on GPU
If a GPU is available, we strongly encourage the users to turn on GPU training by turning on the related json field in
the experiment configs. In all experiments, set  `"cuda": true` (except for the GAN experiments, for which set
`"gpu_mode": true`)
 turn on the "cuda" field in the configurations. This speeds up
training models significantly - especially with Bjorck layers. 
 
### Other Configurations
**Configuring optimizer**: Adam, standard SGD, nesterov momentum and AggMo are supported. Since most of the fields 
in the optimizer configurations are self-explanatory, we leave it for the user to make use of the existing optimizer 
configurations pushed in this repo. 

**Miscellaneous**: Other fields control other aspects of training, such as IO settings, enabling cuda, logging, 
visualizing results etc. 

Other task specific configs will be described below under their corresponding titles. 

# Tasks
Four tasks are explored: Wasserstein Distance estimation, adversarial robustness, GAN training and classification. 

### Wasserstein Distance Estimation
 **Configuring Distributions**: The `distrib1` and `distrib2` fields are intended to be used to configure the probability
distributions that will be used in the Wasserstein Distance estimation tasks. Currently, configs for 
`multi_spherical _shell` (a distribution consisting of multiple spherical shells living in high dimensions) and 
`gan_sampler` (samples from the empirical and generator distribution of a GAN) exist. 
 
 
#### Quantifying Expressivity using Synthetic Distributions
By using synthetic distributions whose Wasserstein distance and its accompanying dual surface we can analytically
compute, we can quantify how expressive a Lipschitz architecture is. The closer the architecture can approximate the 
correct Wasserstein distance, the more expressive it is. 

* Approximating Absolute Value

```
python ./lnets/tasks/dualnets/mains/train_dual.py ./lnets/tasks/dualnets/configs/absolute_value_experiment.json
```

* Approximating Three Cones

```
python ./lnets/tasks/dualnets/mains/train_dual.py ./lnets/tasks/dualnets/configs/three_cones_experiment.json
```

* Approximating High Dimensional Cones

```
python ./lnets/tasks/dualnets/mains/train_dual.py ./lnets/tasks/dualnets/configs/high_dimensional_cone_experiment.json
```

#### Wasserstein Distance between GAN Generator and Empirical Distributions
First, we need to train a GAN so that we can use its generator network for the Wasserstein Distance estimation 
task. 

* GAN training 

(defaults to training WGAN on MNIST)

```
python ./lnets/tasks/gan/mains/train_gan.py ./lnets/tasks/gan/configs/train_GAN.json
```

The GAN type and the training set (along with other training hyperparameters) can be changed:

`gan_type`: One of "WGAN", "WGAN_GP" or "LWGAN" (where the discriminator consists of this paper's contributions - 
more on this later)

`dataset`: One of "mnist", "fashion-mnist", "cifar10", "svhn", "stl10" "lsun-bed"

* Estimating Wasserstein Distance

```
python ./lnets/tasks/dualnets/mains/train_dual.py ./lnets/tasks/dualnets/configs/estimate_wde_gan.json           
```

In order to sample from the GAN trained in the above step, we need to modify the config used for wasserstein distance 
estimation. 

`distrib1.gan_config_json_path`: Path to the gan training config used in the first step. 

One can then modify the model to see which Lipschitz architectures obtain a tighter lower bound on the Wasserstein
distance between the generator and empirical data distribution. 

(warning) Unless the training conditions were exactly the same, the GANs obtained in the GAN training step 
might be slightly different (due to high sensitivity of the training dynamics on initial conditions). Although the estimated 
Wasserstein distances will be different in this case, the relative ordering and approximate ratios of the performance of 
each Lipschitz architectures should be the the same as reported in the paper. We will remedy this by uploading a
trained GAN checkpoint in a future commit. 

### Training LWGAN (Lipschitz WGANs)
We can use the same WGAN training methodology, but build a discriminator network comprised of our methods (i.e. Bjorck 
orthonormalized linear transformations and GroupSort activations)

#### Training LWGAN: 

```
python ./lnets/tasks/gan/mains/train_gan.py ./lnets/tasks/gan/configs/train_LWGAN.json
```

The the training set (along with other training hyperparameters) can be changed:

`dataset`: One of "mnist", "fashion-mnist", "cifar10", "svhn", "stl10" "lsun-bed"

### Classification

#### Classification on Standard Datasets

* Training a standard, fully connected classifier. 

```
python ./lnets/tasks/classification/mains/train_classifier.py ./lnets/tasks/classification/configs/standard/fc_classification.json
```

* Training Bjorck Lipschitz classifier

```
python ./lnets/tasks/classification/mains/train_classifier.py ./lnets/tasks/classification/configs/standard/fc_classification_bjorck.json -o model.linear.bjorck_iter=3
```

Note that we use few bjorck iterations for this training script. Lipschitz-ness will not be strictly enforced so
we do additional finetuning afterwards.

* Orthonormal finetuning

```
python ./lnets/tasks/classification/mains/ortho_finetune.py --model.exp_path=<trained_bjorck_model_path.pt>
```

#### Classification with Small Data
* Generating data indices: Generate which samples in the dataset will be used for training. 

```
python ./lnets/tasks/classification/mains/generate_data_indices.py --data.name mnist --data.root "data/small_mnist" --data.class_count 10 --per_class_count 100 --val_size 5000
```

* Training small data classifier.

```
python ./lnets/tasks/classification/mains/train_classifier.py ./lnets/tasks/classification/configs/small_mnist/lenet_bjorck.json

```

### Adversarial Robustness

For the robustness experiments we trained both the Bjorck orthonormal networks and the L-infinity max-margin networks. In the paper we also compared to the robustness of networks trained without any Lipschitz constraints.

* Training  L-Inf Lipschitz margin network

```
python ./lnets/tasks/classification/mains/train_classifier.py ./lnets/tasks/classification/configs/standard/fc_classification_l_inf_margin.json
```

* Training "standard" network

```
python ./lnets/tasks/classification/mains/train_classifier.py ./lnets/tasks/classification/configs/standard/fc_classification.json
```

* Training with PGD

```
python ./lnets/tasks/adversarial/mains/train_pgd.py ./lnets/tasks/classification/configs/standard/fc_classification.json
```

* Evaluating robustness of trained classifier

```
python ./lnets/tasks/adversarial/mains/manual_eval_adv_robustness.py --model.exp_path="root/of/above/experiment/results" --output_root="outs/adv_robustness/mnist_l_inf_margin"
```

## Code References

* ResNet Implementation: Largely based on github/kuangliu - https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
* GAN training pipeline: Based on and refactored from https://github.com/znxlwm/pytorch-generative-model-collections
