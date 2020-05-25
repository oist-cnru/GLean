# GLean: Goal-directed latent variable inference

This is an implementation of GLean, a goal-directed robot planning and sensory prediction algorithm as described in [Goal-Directed Planning for Habituated Agents by Active Inference Using a Variational Recurrent Neural Network](https://www.mdpi.com/1099-4300/22/5/564).

GLean is based on [PV-RNN](https://arxiv.org/abs/1811.01339) and is capable of generating long horizon fixed length predictions for trained scenarios, given an initial environment state and a goal state. This repository includes sample configuration files and preprocessed datasets in order to reproduce the experimental results in the aforementioned paper.

## Requirements
Branch v2.5:

* Python 2.7 with Numpy, Scipy and Matplotlib
* TensorFlow 1.x

Branch v3.5:

* Python 3.x with Numpy, Scipy and Matplotlib
* TensorFlow >=1.13 (uses tf.compat.v1, 1.15.x recommended)

## Usage
This program is command line and file based. Below is an overview of how to use GLean.

### Preparing the data
All data must be preprocessed using the softmax transform before use. See ops.py for the functions used to accomplish this. Several sample datasets are included.

### Preparing the configuration file
Data and network configurations are stored in cfg files. These configuration files are plain text in the ini format, divided into several sections:

1. \[data\]
* xxxx_path = the path to your npy data file, including subdirectory. Note that the prefix xxxx should match the modality set in the network section that follows. There should be one path per modality
* training_path = the directory to save output as the program runs
* sequences = number of sequences in the npy file. In training this is automatically set, but it is needed for testing and planning as a dummy data file is loaded in those cases
* max_timesteps = seq length in the npy file
* xxxx_dims = number of dimensions of each modality before softmax
* softmax_quant = number of softmax dimensions
* xxxx_softmax_{min, max, sigma} = softmax parameters that were applied to the npy data file

2. \[network\]
* learning_rate = the learning rate (for training and planning)
* gradient_clip = set > 0 to enable gradient clipping (clip_by_global_norm). This also enables pruning of NaN and Inf gradients
* gradient_clip_input = -1 to enable layer normalization, > 0 to clip the input that goes into the neurons (nothing to do with gradients...)
* modalities = comma separated list containing all the modalities. Each modality becomes a "branch" with a shared top layer. Any xxxx prefixed settings must be given for each modality. Note: performance is sub-optimal with multiple modalities
* optimizer = choose the optimizer to use: adam (recommended), gradient_descent, momentum, adagrad, rmsprop, rmsprop_momentum
* xxxx_celltype = choose the neuron model to use: MTRNN, LSTM
* xxxx_activation_func = choose the activation function used used in the neuron: tanh (recommended), relu, leaky_relu, sigmoid, extended_tanh
* xxxx_meta_prior = comma separated list containing the meta prior values for each layer
* xxxx_layers_neurons = comma separated list containing the number of neurons per layer
* xxxx_layers_param = comma separated list containing the timescale per layer (MTRNN) or forget bias (LSTM)
* xxxx_layers_z_units = comma separted list containing the number of z units per layer. Keep this small for performance

3. \[training\]
Optional section for some additional settings during training
* max_epochs = set the maximum number of epochs to run for in training
* fig_xy_dims = a comma separated list of dimensions to plot in a XY diagram. Should be exactly two dimension indexes, in any order. Leave this unset to plot all dimensions with respect to timestep
* fig_{x,y}_lim = two values, comma separated, representing the X and Y range in the plot

4. \[testing\]
Similar to the training section. See the section on operation.

5. \[planning\]
Unlike the testing and training sections, this section is required for plan generation
* max_epochs = set the maximum number of epochs to run for in planning
* init_modalities = what modalities are provided in the initial frame(s)
* goal_modalities = what modalities are provided in the goal frame(s)
* goal_modalities_mask = two values, comma separated, that represent the range of dimensions (pre-softmax) that are given in the goal frame(s). For example, some joint angles can be masked out to reduce the amount of ground truth data that is given to the planner

### Operation
There are three steps in order use GLean: train a network, test its output and generate a plan. The network must be trained first, which requires a configuration file as described previously. Once a network has been trained, it should be tested to see if it can generate the learned patterns, either using posterior regeneration (default) or prior generation. Finally, plan generation can be done using untrained test data. 

As an example, executing the following commands in this directory will train a network, test the prior generation after training, and then do plan generation with test data:

1. python train.py config/branching_path_2g_60.cfg
2. python test.py config/branching_path_2g_60.cfg --prior_only
3. python plan.py config/branching_path_2g_60.cfg --motor_datapath data/branching_path/test/path_2g_test_0_softmax10.npy

Output can be found in the _output_ subdirectory.

Many of the configuration options can be overridden with command line arguments. Some important command line arguments are:

* View available command line arguments: --help
* Choose which training checkpoint to use: --checkpoint _model-yyyy_
* Override the training path: --training_path _path_
* Override the path to the data file - required for pointing to test data during planning: --xxxx_datapath _path_ (Note: in this repository, the 2D dataset uses the prefix _motor_, while the toroboarmsim dataset uses the prefix _robot_)
* Testing only - use prior generation (otherwise do posterior regeneration): --prior_only
* Testing and planning only - change the subdirectory output is stored in: --output_dir _path_
