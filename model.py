import numpy as np
import tensorflow as tf
import functools
import sys
import os
import re
from tensorflow.python.framework import ops
from tensorflow.python.ops import functional_ops

from BasicRNNCell import BasicLSTMCell as LSTM
from BasicRNNCell import BasicMTRNNCell as MTRNN
from BasicRNNCell import _linear
import ops

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pkg_resources import parse_version

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def print_dict(dict):
    for key in dict:
        print(str(key) + " = " + str(dict[key]))

"""
Predictive Coding Variational Bayes Recurrent Neural Network (PV-RNN)
"""
class PVRNN(object):
    def __init__(self, sess, config_data, config_network, training=True, planning=None, input_cl_ratio=1.0, learning_rate=0.001, optimizer_epsilon=1e-8, prior_generation=False, reset_posterior_src=False, hybrid_posterior_src=None, hybrid_posterior_src_idx=None, data_masking=False, overrides=dict()):
        self.sess = sess
        self.modalities = [str(m.strip()) for m in config_network["modalities"].split(',')] # determine how many modalities are used

        ## Dump config
        print("training = " + str(training))
        if training:
            print("**learning args**")
            print("learning_rate = " + str(learning_rate))
            print("optimizer_epsilon = " + str(optimizer_epsilon))
        print("**generation args**")
        print("prior_generation = " + str(prior_generation))
        print("reset_posterior_src = " + str(reset_posterior_src))
        print("hybrid_posterior_src = " + str(hybrid_posterior_src))
        print("hybrid_posterior_src_idx = " + str(hybrid_posterior_src_idx))
        print("**config_data**")
        print_dict(config_data)
        print("**config_network**")
        print_dict(config_network)
        if planning is not None:
            print("**planning**")
            print_dict(planning)
        if overrides:
            print("**overrides**")
            print_dict(overrides)

        ## Data
        self.n_seq = int(config_data["sequences"]) # number of sequences
        self.mask_seq = data_masking # use masking to compensate for different sequence lengths (disable if data is padded)
        self.max_timesteps = int(config_data["max_timesteps"]) # if real data is loaded, this is overwritten
        self.dims = dict()
        self.path = dict()
        for m in self.modalities:
            self.dims[m] = int(config_data[m + "_dims"]) 
            self.path[m] = config_data[m + "_path"]
        self.batch_size = self.n_seq # = n for performance, = 1 for online learning/error regression
        self.softmax_quant = int(config_data["softmax_quant"])
        self.override_load_data = True if overrides is not None and "load_training_data" in overrides and overrides["load_training_data"] else False

        ## Training
        self.training = training # set whether the network is learning (training or planning)
        self.planning = True if planning is not None else False # set to do error regression to generate a plan from start to goal
        if self.planning:
            self.planning_init_modalities = [x.strip() for x in planning["init_modalities"].split(',')] if "init_modalities" in planning else self.modalities[0]
            self.planning_goal_modalities = [x.strip() for x in planning["goal_modalities"].split(',')]  if "goal_modalities" in planning else self.modalities[0]
            self.planning_goal_modalities_mask = [int(x.strip()) for x in planning["goal_modalities_mask"].split(',')] if "goal_modalities_mask" in planning else None
            self.planning_init_modalities_mask = [int(x.strip()) for x in planning["init_modalities_mask"].split(',')] if "init_modalities_mask" in planning else None

        ## Planning
        self.planning_initial_frame = 0 # start of initial frames
        self.planning_initial_depth = 1 # how many frames to provide at the start
        self.planning_duplicate_initial_frame = False # set to true to duplicate initial frame for padding
        self.planning_goal_frame = -1 # start of goal frames (should be negative index)
        self.planning_goal_depth = None # how many frames to provide at the end (set to None to continue until the end of the sequence)
        self.planning_goal_offset = 0 # move goal position in plan relative to target data
        self.planning_goal_padding = False # set to true to copy the goal frame until the end of the sequence
        self.planning_auto_weight = 1 # increase rec error pressure by a factor of missing frames (0 to disable, >1 to multiply)

        if self.planning:
            if "init_frame" in planning:
                self.planning_initial_frame = int(planning["init_frame"])
            if "init_depth" in planning:
                self.planning_initial_depth = int(planning["init_depth"]) if planning["init_depth"].lower() != "none" else None
            if "goal_frame" in planning:
                self.planning_goal_frame = int(planning["goal_frame"])
            if "goal_depth" in planning:
                self.planning_goal_depth = int(planning["goal_depth"]) if planning["goal_depth"].lower() != "none" else None
            if "init_frame_duplicate" in planning:
                self.planning_duplicate_initial_frame = True if planning["init_frame_duplicate"].lower() == "true" else False
            if "goal_padding" in planning:
                self.planning_goal_padding = True if planning["goal_padding"].lower() == "true" else False
            if "rec_weighting" in planning:
                self.planning_auto_weight = float(planning["rec_weighting"])

        ## Optimizer
        if "optimizer" in config_network:
            self.optimizer_func = config_network["optimizer"].lower()
        else:
            self.optimizer_func = "adam"
            print("model: Using default optimizer adam")
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.gradient_clip = float(config_network["gradient_clip"])

        ## Model
        self.d_neurons = dict()
        self.z_units = dict()
        self.n_layers = dict()
        self.layers = dict()
        self.layers_names = dict()
        self.layers_params = dict()
        self.shared_layer = None
        # Default connections
        self.connect_z = True
        self.connect_topdown_dz = False
        self.connect_topdown_dt = False
        self.connect_horizontal = True # connect between modalities
        self.connect_topdown_d = True # connect from higher to lower layers
        self.connect_bottomup_d = True # connect from lowest to higher layers
        # Overrides
        if "connect_z" in config_network:
            self.connect_z = True if config_network["connect_z"].lower() == "true" else False
        if "connect_topdown_dz" in config_network:
            self.connect_topdown_dz = True if config_network["connect_topdown_dz"].lower() == "true" else False
        if "connect_topdown_dt" in config_network:
            self.connect_topdown_dt = True if config_network["connect_topdown_dt"].lower() == "true" else False
        if "connect_horizontal" in config_network:
            self.connect_horizontal = True if config_network["connect_horizontal"].lower() == "true" else False
        if "connect_topdown_d" in config_network:
            self.connect_topdown_d = True if config_network["connect_topdown_d"].lower() == "true" else False
        if "connect_bottomup_d" in config_network:
            self.connect_bottomup_d = True if config_network["connect_bottomup_d"].lower() == "true" else False

        self.layers_concat_input = False # use concatenation instead of addition when preparing input to neurons
        self.gradient_clip_input = float(config_network["gradient_clip_input"]) # clip layer output values
        self.dropout_mask_error = False if not self.training or (overrides is not None and "dropout_mask_error" in overrides and not overrides["dropout_mask_error"]) else True # use masking to manipulate reconstruction error in training and planning
        if "override_d_output" in overrides and overrides["override_d_output"] is not None:
            self.override_d_output = overrides["override_d_output"][1] # set to enable overriding the output of D per timestep with desired value, starting with output layer (L0)
            self.override_d_output_range = overrides["override_d_output"][0]
        else:
            self.override_d_output = None
            self.override_d_output_range = None
        if "kld_range" in overrides:
            self.override_kld_range = [int(x.strip()) for x in overrides["kld_range"].split(',')]
        else:
            self.override_kld_range = None

        for m in self.modalities:
            self.d_neurons[m] = [int(x.strip()) for x in config_network[m + "_layers_neurons"].split(',')]
            if m + "_layers_z_units" in config_network:
                self.z_units[m] = [int(x.strip()) for x in config_network[m + "_layers_z_units"].split(',')]
            else:
                self.z_units[m] = [int(round(float(d)/10)) for d in self.d_neurons[m]]
                print("model: Using default z_units " + str(self.z_units[m]))
            if m + "_layers_param" in config_network:
                self.layers_params[m] = [float(x.strip()) for x in config_network[m + "_layers_param"].split(',')]
            else:
                self.layers_params[m] = [float(2**(i+1)) for i in range(len(self.d_neurons[m]))]
                print("model: Using default layer parameters " + str(self.layers_params[m]))
            # Append layer 0
            self.d_neurons[m].insert(0, (self.dims[m] * self.softmax_quant))
            self.z_units[m].insert(0, 0)
            self.layers[m] = [None for _ in range(len(self.d_neurons[m]))] # Layer 0 is for output, no cells
            self.layers_names[m] = ["l" + str(l) + "_" + m for l in range(len(self.d_neurons[m]))]
            self.n_layers[m] = len(self.layers[m]) # including I/O

        # Assume only the top layer might be shared
        if max(self.n_layers.values()) != min(self.n_layers.values()):
            self.shared_layer = max(self.n_layers, key=self.n_layers.get)

        ## Variational Bayes
        self.vb_meta_prior = dict()
        self.vb_seq_prior = dict()
        for m in self.modalities:
            self.vb_meta_prior[m] = [float(x.strip()) for x in config_network[m + "_meta_prior"].split(',')] # meta-prior setting
            if m + "_seq_prior" in config_network:
                self.vb_seq_prior[m] = [False if x.strip().lower() == "false" else True for x in config_network[m + "_seq_prior"].split(',')]
                if len(self.vb_seq_prior[m]) < self.n_layers[m]-1:
                    self.vb_seq_prior[m] = [self.vb_seq_prior[m][0]] * (self.n_layers[m]-1)
                    print("model: Assuming all layers of " + m + " to be sequential prior=" + str(self.vb_seq_prior[m][0]))
            else:
                self.vb_seq_prior[m] = [True] * (self.n_layers[m]-1) # set to false to use unit gaussian in Z calculation
            self.vb_seq_prior[m].insert(0, False) # I/O layer
        if "ugaussian_t_range" in config_network:
            self.vb_ugaussian_t_range = [int(x.strip()) for x in config_network["ugaussian_t_range"].split(',')]
            if "ugaussian_weight" in config_network: # larger weight = less initial sensitivity
                self.vb_ugaussian_weight = float(config_network["ugaussian_weight"])
            else:
                self.vb_ugaussian_weight = 0.001
        else:
            self.vb_ugaussian_t_range = None
            self.vb_ugaussian_weight = None
        self.vb_new_meta_prior_loss = True # use new loss calculation
        self.vb_return_full_loss = config_network.get("return_full_loss", False) # returns loss per timestep per sequence (true or false, not text)
        self.vb_per_t_meta_prior = False # apply W at loss calculation per timestep (always true when vb_ugaussian is used)
        self.vb_zero_initial_out = True # set to true for d=0 at t=0
        self.vb_reset_posterior = reset_posterior_src # reset the posterior's trained input, use in planning
        self.vb_prior_output = prior_generation # use either prior or posterior in calculating output
        self.vb_posterior_past_input = True if config_network.get("connect_posterior_dz", "false").lower() == "true" else False # set to true to include d_{t-1} in posterior calculation
        self.vb_posterior_src_extend = True if config_network.get("posterior_map_src", "false").lower() == "true" else False # set to true apply weights to Z source
        self.vb_posterior_blend_factor = 0.0 # combine posterior and prior during plan generation
        self.vb_limit_sigma = False # hard clip sigma to be non-zero and not too large
        self.vb_hybrid_posterior_src = False if hybrid_posterior_src is None else True # create a window of trained A values
        self.vb_hybrid_posterior_src_range = hybrid_posterior_src # how many trained A samples to provide
        self.vb_hybrid_posterior_src_zero_init = False # set to true to zero A in the window instead of providing a trained A window
        self.vb_hybrid_posterior_src_idx_override = None if hybrid_posterior_src_idx is None else np.repeat(hybrid_posterior_src_idx, self.n_seq)
        self.vb_hybrid_prior_override = overrides.get("hybrid_posterior_override", False) # set to true to use posterior for all Z calculations during window

        self.vb_prior_override_l = None
        self.vb_prior_override_t_range = None
        self.vb_prior_override_sigma = None
        self.vb_prior_override_myu = None
        self.vb_prior_override_epsilon = None
        if "prior_override_l" in overrides:
            # set to None, True or a list of levels to override (starting at 1)
            if overrides["prior_override_l"].lower() == "all" or overrides["prior_override_l"].lower() == "true":
                self.vb_prior_override_l = True
            elif overrides["prior_override_l"].lower() != "none" or overrides["prior_override_l"].lower() != "false":
                self.vb_prior_override_l = [int(x.strip()) for x in overrides["prior_override_l"].split(',')]

            if "prior_override_t_range" in overrides and overrides["prior_override_t_range"] is not None:
                self.vb_prior_override_t_range = [int(x.strip()) for x in overrides["prior_override_t_range"].split(',')]

            self.vb_prior_override_sigma = float(overrides["prior_override_sigma"]) if "prior_override_sigma" in overrides else None
            self.vb_prior_override_myu = float(overrides["prior_override_myu"]) if "prior_override_myu" in overrides else None
            self.vb_prior_override_epsilon = float(overrides["prior_override_epsilon"]) if "prior_override_epsilon" in overrides else None # set to 0 to disable noise sampling

        self.vb_posterior_override_l = None
        self.vb_posterior_override_t_range = None
        self.vb_posterior_override_sigma = None
        self.vb_posterior_override_myu = None
        self.vb_posterior_override_epsilon = None
        if "posterior_override_l" in overrides:
            # set to None, True or a list of levels to override (starting at 1)
            if overrides["posterior_override_l"].lower() == "all" or overrides["posterior_override_l"].lower() == "true":
                self.vb_posterior_override_l = True
            elif overrides["posterior_override_l"].lower() != "none" or overrides["posterior_override_l"].lower() != "false":
                self.vb_posterior_override_l = [int(x.strip()) for x in overrides["posterior_override_l"].split(',')]

            if "posterior_override_t_range" in overrides and overrides["posterior_override_t_range"] is not None:
                self.vb_posterior_override_t_range = [int(x.strip()) for x in overrides["posterior_override_t_range"].split(',')]

            self.vb_posterior_override_sigma = float(overrides["posterior_override_sigma"]) if "posterior_override_sigma" in overrides else None
            self.vb_posterior_override_myu = float(overrides["posterior_override_myu"]) if "posterior_override_myu" in overrides else None
            self.vb_posterior_override_epsilon = float(overrides["posterior_override_epsilon"]) if "posterior_override_epsilon" in overrides else None # set to 0 to disable noise sampling

        # Activation functions
        self.activation_func = dict()
        self.z_activation_func = dict()
        for m in self.modalities:
            # Supported activation functions: ReLU, Leaky ReLU, Sigmoid, Extended Hyperbolic Tangent, Hyperbolic Tangent (default)
            if m + "_activation_func" in config_network:
                if config_network[m + "_activation_func"].lower() == "relu":
                    self.activation_func[m] = tf.nn.relu
                elif config_network[m + "_activation_func"].lower() == "leaky_relu":
                    self.activation_func[m] = tf.nn.leaky_relu
                elif config_network[m + "_activation_func"].lower() == "sigmoid":
                    self.activation_func[m] = tf.nn.sigmoid
                elif config_network[m + "_activation_func"].lower() == "extended_tanh":
                    self.activation_func[m] = ops.extended_hyperbolic
                elif config_network[m + "_activation_func"].lower() == "tanh":
                    self.activation_func[m] = tf.nn.tanh
                else:
                    self.activation_func[m] = tf.nn.tanh
                    print("model: Unknown activation function " + config_network[m + "_activation_func"] + ", falling back to tanh")
            else:
                self.activation_func[m] = tf.nn.tanh
                print("model: Using default activation function tanh")
            self.z_activation_func[m] = self.activation_func[m]

            # Layer 1+
            for i in range(1,self.n_layers[m]):
                with tf.compat.v1.variable_scope(self.layers_names[m][i]):
                    # Supported celltypes: LSTM and MTRNN (default)
                    if m + "_celltype" in config_network:
                        if config_network[m + "_celltype"].lower() == "lstm":
                            self.layers[m][i] = LSTM(self.d_neurons[m][i], activation=self.activation_func[m], forget_bias=self.layers_params[m][i-1])
                        elif config_network[m + "_celltype"].lower() == "mtrnn":
                            self.layers[m][i] = MTRNN(self.d_neurons[m][i], activation=self.activation_func[m], tau=self.layers_params[m][i-1])
                        else:
                            self.layers[m][i] = MTRNN(self.d_neurons[m][i], activation=self.activation_func[m], tau=self.layers_params[m][i-1])
                            print("model: Unknown cell type " + config_network[m + "_celltype"] + ", falling back to MTRNN")
                    else:
                        self.layers[m][i] = MTRNN(self.d_neurons[m][i], activation=self.activation_func[m], tau=self.layers_params[m][i-1])
                        print("model: Using default cell type MTRNN")
        
        # Layer 0
        self.input_provide_data = False # set to false to not provide any input to the network, causes cl_ratio to be ignored
        self.input_cl_ratio = input_cl_ratio
        self.output_z_factor = dict()
        for m in self.modalities:
            if m + "_output_z_factor" in config_network: # how much L1 z-units influence output for additional regularization (0.0 = disabled)
                self.output_z_factor[m] = float(config_network[m + "_output_z_factor"])
            else:
                self.output_z_factor[m] = 0.0

        self.build_model
    # __init__ #

    ## For backwards compatibility, data must be preprocessed into the P-DVMRNN npy format first
    def load_dataset(self):
        data_raw = dict()
        if self.planning:
            data_orig_raw = dict()
        for m in self.modalities:
            ## Motor data
            if (self.training and not self.planning) or self.override_load_data:
                data_raw[m] = np.load(self.path[m]) # seq, timestep, dim (softmax)
                print("load_dataset: Loaded data for " + m + " with shape " + str(np.shape(data_raw[m])))
            else:
                data_raw[m] = np.zeros((self.n_seq, self.max_timesteps, self.dims[m]*self.softmax_quant))
                print("load_dataset: Generated null dataset for " + m + " with shape " + str(np.shape(data_raw[m])))

            if self.planning:
                data_orig_raw[m] = np.load(self.path[m]) # seq, timestep, dim (softmax)
                print("load_dataset: Loaded ground truth data for " + m + " with shape " + str(np.shape(data_orig_raw[m])))

                plan_goal_start = self.planning_goal_frame + self.planning_goal_offset
                plan_goal_end = None if self.planning_goal_depth is None else plan_goal_start + self.planning_goal_depth + self.planning_goal_offset

                load_init_end = self.planning_initial_frame+self.planning_initial_depth if not self.planning_duplicate_initial_frame else self.planning_initial_frame+1
                load_goal_end = None if self.planning_goal_depth is None else plan_goal_start + self.planning_goal_depth

                # Copy initial frame(s)
                if self.planning_init_modalities_mask is not None:
                    plan_mask_start = self.planning_init_modalities_mask[0]*self.softmax_quant
                    plan_mask_end = (self.planning_init_modalities_mask[1]+1)*self.softmax_quant if self.planning_init_modalities_mask[1] is not None else None
                    if not self.planning_duplicate_initial_frame:
                        data_raw[m][:, self.planning_initial_frame:self.planning_initial_depth, plan_mask_start:plan_mask_end] = data_orig_raw[m][:, self.planning_initial_frame:load_init_end, plan_mask_start:plan_mask_end]
                    else: # only mask duplicated frames
                        data_raw[m][:, self.planning_initial_frame, :] = data_orig_raw[m][:, self.planning_initial_frame, :]
                        data_raw[m][:, self.planning_initial_frame+1:self.planning_initial_depth, plan_mask_start:plan_mask_end] = data_orig_raw[m][:, self.planning_initial_frame:load_init_end, plan_mask_start:plan_mask_end]
                else:
                    data_raw[m][:, self.planning_initial_frame:self.planning_initial_depth, :] = data_orig_raw[m][:, self.planning_initial_frame:load_init_end, :]

                # Copy goal frame(s)
                if self.planning_goal_modalities_mask is not None:
                    plan_mask_start = self.planning_goal_modalities_mask[0]*self.softmax_quant
                    plan_mask_end = (self.planning_goal_modalities_mask[1]+1)*self.softmax_quant if self.planning_goal_modalities_mask[1] is not None else None
                    data_raw[m][:, plan_goal_start:plan_goal_end, plan_mask_start:plan_mask_end] = data_orig_raw[m][:, self.planning_goal_frame:load_goal_end, plan_mask_start:plan_mask_end] # copy goal frame
                else:
                    data_raw[m][:, plan_goal_start:plan_goal_end, :] = data_orig_raw[m][:, self.planning_goal_frame:load_goal_end, :] # copy goal frame
                if self.planning_goal_padding:
                    data_raw[m][:, plan_goal_end:, :] = data_orig_raw[m][:, load_goal_end, :] # duplicate last frame
        return self._load_dataset(data_raw)

    ## Using Dataset API
    def _load_dataset(self, data_raw):
        data_tensors = dict()
        data_dataset = dict()
        setlist = []
        batch = dict()

        for m in self.modalities:
            data_tensors[m] = tf.convert_to_tensor(value=data_raw[m], dtype=tf.float32, name="load_data_tensors")  # [seq, step, dim*quant_level]
            data_dataset[m] = tf.data.Dataset.from_tensor_slices(data_tensors[m])
        
        data_shape = np.shape(next(iter(data_raw.values())))
        # Override defined number of sequences with real value
        if self.n_seq == self.batch_size or data_shape[0] < self.batch_size:
            self.batch_size = data_shape[0] # update batch size
        self.n_seq = data_shape[0]
        if self.n_seq < self.batch_size:
            self.batch_size = data_shape[0] # fix batch size

        # Override predefined number of timesteps with real value
        self.max_timesteps = data_shape[1]

        # Build index
        idxs = [[j for _ in range(self.max_timesteps)] for j in range(self.n_seq)]
        idx_tensors = tf.convert_to_tensor(value=idxs, dtype=tf.int32, name="load_idx_tensors")
        idx_dataset = tf.data.Dataset.from_tensor_slices(idx_tensors)

        # Batching
        setlist.append(idx_dataset)
        for m in self.modalities:
            setlist.append(data_dataset[m])
        dataset = tf.data.Dataset.zip(tuple(setlist))

        if self.training and not self.planning:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=self.n_seq//self.batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

        batch = iterator.get_next()

        out = dict()
        out["idx_data"] = idx_tensors
        out["idx_next"] = batch[0]
        for i in range(len(self.modalities)):
            out[self.modalities[i] + "_data"] = data_tensors[self.modalities[i]]
            out[self.modalities[i] + "_next"] = batch[i+1]
        out["timesteps"] = self.max_timesteps
        return out

    @lazy_property
    def build_model(self):
        ## Dataset
        data = self.load_dataset()
        if not self.mask_seq:
            data_mask_reg = None
        else:
            data_mask_reg = ops.data_mask(data[self.modalities[0] + "_next"]) # mask for different sequence lengths
        if not self.input_provide_data:
            data_mask_rec = data_mask_reg
        else:
            data_mask_rec = ops.data_mask(data[self.modalities[0] + "_next"], skip_ahead=1)

        ## Initialize initial states for all network units
        with tf.compat.v1.variable_scope('initial_state', reuse=tf.compat.v1.AUTO_REUSE):
            init_data = data if self.input_provide_data else None
            initial_states = ops.set_trainable_initial_states(self.modalities, init_data, self.batch_size, self.d_neurons, self.z_units)

        ## Train model
        with tf.compat.v1.variable_scope("training"):
            data_next = []
            data_next.append(tf.transpose(a=data["idx_next"], perm=[1, 0], name="transpose_idx_next"))
            if self.input_provide_data:
                for m in self.modalities:
                    data_next.append(tf.transpose(a=data[m + "_next"], perm=[1, 0, 2], name="transpose_data_next")) # step, batch, dim
            step_data = tuple(data_next)

            # Run one epoch, all timesteps
            output_model = functional_ops.scan(self.build_model_one_step_scan, step_data, initializer=initial_states, parallel_iterations=self.batch_size)

            # Collect output (dicts)
            generated_out = {m: tf.transpose(a=output_model["out"][m][0], perm=[1, 0, 2], name="transpose_generated_out") for m in self.modalities} # output layer 0: seq, step, dim
            generated_z_prior = dict()
            generated_z_prior_mean = dict()
            generated_z_prior_var = dict()
            generated_z_posterior = dict()
            generated_z_posterior_mean = dict()
            generated_z_posterior_var = dict()
            generated_z_posterior_src = dict()
            for m in self.modalities:
                if max(self.vb_meta_prior[m]) >= 0:
                    generated_z_prior[m] = [tf.transpose(a=output_model["z_prior"][m][i], perm=[1, 0, 2], name="transpose_generated_z_p") for i in range(len(output_model["z_prior"][m]))]
                    generated_z_prior_mean[m] = [tf.transpose(a=output_model["z_prior_mean"][m][i], perm=[1, 0, 2], name="transpose_generated_zm_p") for i in range(len(output_model["z_prior_mean"][m]))]
                    generated_z_prior_var[m] = [tf.transpose(a=output_model["z_prior_var"][m][i], perm=[1, 0, 2], name="transpose_generated_zv_p") for i in range(len(output_model["z_prior_var"][m]))]
                    generated_z_posterior[m] = [tf.transpose(a=output_model["z_posterior"][m][i], perm=[1, 0, 2], name="transpose_generated_z_q") for i in range(len(output_model["z_posterior"][m]))]
                    generated_z_posterior_mean[m] = [tf.transpose(a=output_model["z_posterior_mean"][m][i], perm=[1, 0, 2], name="transpose_generated_zm_q") for i in range(len(output_model["z_posterior_mean"][m]))]
                    generated_z_posterior_var[m] = [tf.transpose(a=output_model["z_posterior_var"][m][i], perm=[1, 0, 2], name="transpose_generated_zv_q") for i in range(len(output_model["z_posterior_var"][m]))]

                    if any(self.vb_seq_prior[m]):
                        with tf.compat.v1.variable_scope("model_variables", reuse=True):
                            src_z = max(self.z_units[m]) if self.vb_posterior_src_extend else max(self.z_units[m])*2
                            if not self.vb_hybrid_posterior_src:
                                if not self.vb_reset_posterior:
                                    z_posterior_src_var_name = m + "_z_posterior_src"
                                else:
                                    z_posterior_src_var_name = m + "_z_posterior_src_zero"
                                generated_z_posterior_src[m] = tf.compat.v1.get_variable(z_posterior_src_var_name, shape=[self.max_timesteps, self.n_layers[m], self.n_seq, src_z])
                            else:
                                trained_src = tf.compat.v1.get_variable(m + "_z_posterior_src", shape=[self.max_timesteps, self.n_layers[m], self.n_seq, src_z])
                                zero_src = tf.compat.v1.get_variable(m + "_z_posterior_src_zero", shape=[self.max_timesteps, self.n_layers[m], self.n_seq, src_z])
                                initial_start = self.vb_hybrid_posterior_src_range[0]
                                initial_end = self.vb_hybrid_posterior_src_range[1]
                                if not self.vb_hybrid_posterior_src_zero_init:
                                    generated_z_posterior_src[m] = tf.concat([trained_src[initial_start:initial_end, :, :, :], zero_src[initial_end:, :, :, :]], axis=0, name="concat_hybrid_src_tz")
                                else:
                                    generated_z_posterior_src[m] = tf.concat([zero_src[initial_start:initial_end, :, :, :], trained_src[initial_end:, :, :, :]], axis=0, name="concat_hybrid_src_zt")

        ## Calculate loss per modality
        batch_reconstruction_loss = dict.fromkeys(self.modalities, tf.constant(0.0))
        batch_regularization_loss = dict.fromkeys(self.modalities, tf.constant(0.0))
        for m in self.modalities:
            ## Reconstruction loss
            if not self.planning:
                if not self.dropout_mask_error:
                    rec_loss = ops.kld_with_mask(data[m + "_next"][:, :, :], generated_out[m][:, :data["timesteps"], :], data_mask_rec)
                    if self.vb_return_full_loss:
                        batch_reconstruction_loss[m] = (rec_loss[0] / float(self.dims[m]) / float(self.softmax_quant), rec_loss[1] / float(self.dims[m]) / float(self.softmax_quant))
                    else:
                        batch_reconstruction_loss[m] = (rec_loss[0] / float(self.dims[m]) / float(self.softmax_quant), rec_loss[1])
                else:
                    dropout_mask = ops.dropout_mask([self.batch_size, self.max_timesteps])
                    rec_loss = ops.kld_with_mask(data[m + "_next"][:, :, :], generated_out[m][:, :data["timesteps"], :], dropout_mask)
                    if self.vb_return_full_loss:
                        batch_reconstruction_loss[m] = (rec_loss[0] * 2.0 / float(self.dims[m]) / float(self.softmax_quant), rec_loss[1] * 2.0 / float(self.dims[m]) / float(self.softmax_quant))
                    else:
                        batch_reconstruction_loss[m] = (rec_loss[0] * 2.0 / float(self.dims[m]) / float(self.softmax_quant), rec_loss[1])
            else: # Rec loss only exists at initial and goal frames
                selected_loss_idx = -1
                selected_loss = tf.constant(sys.float_info.max)
                selected_loss_rec = selected_loss
                selected_loss_reg = selected_loss
                if m in self.planning_init_modalities:
                    plan_iframe_start = self.planning_initial_frame
                    plan_iframe_end = self.planning_initial_frame + self.planning_initial_depth
                if m in self.planning_goal_modalities:
                    plan_gframe_start = self.planning_goal_frame + self.planning_goal_offset
                    plan_gframe_end = None if self.planning_goal_depth is None else plan_gframe_start + self.planning_goal_depth + self.planning_goal_offset
                    if self.planning_goal_modalities_mask is not None:
                        plan_mask_start = self.planning_goal_modalities_mask[0]*self.softmax_quant
                        plan_mask_end = (self.planning_goal_modalities_mask[1]+1)*self.softmax_quant if self.planning_goal_modalities_mask[1] != -1 else -1
                    else:
                        plan_mask_start = None
                        plan_mask_end = None
                error_weight = 1.0
                if m not in self.planning_init_modalities and m not in self.planning_goal_modalities:
                    batch_reconstruction_loss[m] = 0.0
                else:
                    if m in self.planning_init_modalities and self.planning_auto_weight > 0:
                        error_weight += self.planning_initial_depth
                    if m in self.planning_goal_modalities:
                        if plan_mask_end == -1:
                            plan_mask_end = self.dims[m]*self.softmax_quant
                        if self.planning_auto_weight > 0:
                            error_weight += 1.0
                    if self.planning_auto_weight > 0:
                        error_weight = (self.max_timesteps * self.planning_auto_weight) / error_weight
                    if self.planning_goal_modalities_mask is not None:
                        dmask1 = [1 if d >= plan_mask_start and d < plan_mask_end else 0 for d in range(self.dims[m]*self.softmax_quant)]
                        planning_mask = ops.windowed_dmask(dmask1, [self.n_seq, self.max_timesteps, self.dims[m]*self.softmax_quant], start=[plan_iframe_start, plan_iframe_end], end=[plan_gframe_start, plan_gframe_end], end_zeropad=(not self.planning_goal_padding))
                        rec_loss = ops.kld_with_mask(data[m + "_next"][:, :, :], generated_out[m][:, :data["timesteps"], :], dmask=planning_mask)
                        batch_reconstruction_loss[m] = rec_loss # (reduced, all sequences)
                        error_weight += 1.0
                    else:
                        planning_mask = ops.windowed_mask([self.batch_size, self.max_timesteps], start=[plan_iframe_start, plan_iframe_end], end=[plan_gframe_start, plan_gframe_end], end_zeropad=(not self.planning_goal_padding))
                        rec_loss = ops.kld_with_mask(data[m + "_next"][:, :, :], generated_out[m][:, :data["timesteps"], :], mask=planning_mask)
                        batch_reconstruction_loss[m] = rec_loss # (reduced, all sequences)

                    # Average
                    batch_reconstruction_loss[m] = (batch_reconstruction_loss[m][0] * error_weight / float(self.dims[m]) / self.softmax_quant, batch_reconstruction_loss[m][1] * error_weight / float(self.dims[m]) / self.softmax_quant)

            ## Regularization loss (per layer)
            if max(self.vb_meta_prior[m]) >= 0:
                if self.override_kld_range is not None:
                    # Override mask
                    reg_mask = ops.windowed_mask([self.batch_size, self.max_timesteps], start=self.override_kld_range, end=[0,0])
                else:
                    reg_mask = data_mask_reg
                batch_regularization_loss[m] = ops.vb_kld_with_mask(output_model, reg_mask, self.z_units[m], m, self.vb_seq_prior[m], self.vb_ugaussian_t_range, self.vb_ugaussian_weight, self.vb_meta_prior[m], seq_kld_weight_by_t=self.vb_per_t_meta_prior)

        ## Calculate total loss
        total_batch_loss = tf.constant(0.0)
        total_batch_reconstruction_loss = tf.constant(0.0)
        total_batch_regularization_loss = tf.constant(0.0)
        # Find least loss for planner
        if self.planning:
            full_batch_reconstruction_loss = tf.zeros(tf.shape(input=batch_reconstruction_loss[m][1])[0])
            full_batch_regularization_loss = tf.zeros_like(full_batch_reconstruction_loss)
            full_batch_loss = tf.zeros_like(full_batch_reconstruction_loss)
        for m in self.modalities:
            # Reconstruction loss
            rec_loss = batch_reconstruction_loss[m][0]
            total_batch_reconstruction_loss += rec_loss
            if self.planning:
                rec_loss_seq = tf.reduce_sum(input_tensor=batch_reconstruction_loss[m][1], axis=1, name="reduce_recloss")
                full_batch_reconstruction_loss += rec_loss_seq
            # Regularization loss
            if self.vb_meta_prior[m][0] != -1:
                zs = len(self.z_units[m])-1
                for i in range(1, zs+1):
                    reg_loss = batch_regularization_loss[m][i][0]
                    total_batch_regularization_loss += reg_loss
                    if self.vb_ugaussian_t_range is not None or self.vb_per_t_meta_prior: # W is applied in KLD calculation
                        total_batch_loss += rec_loss/zs - reg_loss
                    else: # apply W to the whole sequence
                        W = self.vb_meta_prior[m][i-1] if self.vb_meta_prior[m][i-1] >= 0 else 0.0
                        if self.vb_new_meta_prior_loss:
                            total_batch_loss += rec_loss/zs - W * reg_loss
                        else:
                            total_batch_loss += ((1.0 - W) * (rec_loss/zs)) - (W * reg_loss)
                
                if self.planning:
                    for i in range(1, zs+1):
                        reg_loss_seq = tf.reduce_sum(input_tensor=batch_regularization_loss[m][i][1], axis=1, name="reduce_regloss")
                        full_batch_regularization_loss += reg_loss_seq
                        if self.vb_ugaussian_t_range is not None or self.vb_per_t_meta_prior: # W is applied in KLD calculation
                            full_batch_loss += rec_loss_seq/zs - reg_loss_seq
                        else: # apply W to the whole sequence
                            W = self.vb_meta_prior[m][i-1] if self.vb_meta_prior[m][i-1] >= 0 else 0.0
                            if self.vb_new_meta_prior_loss:
                                full_batch_loss += rec_loss_seq/zs - W * reg_loss_seq
                            else:
                                full_batch_loss += ((1.0 - W) * (rec_loss_seq/zs)) - (W * reg_loss_seq)
            else:
                total_batch_loss = total_batch_reconstruction_loss

            # Select least loss as recommended plan
            if self.planning:            
                selected_loss_idx = tf.argmin(input=full_batch_loss, name="argmin_full_loss")
                selected_loss = full_batch_loss[selected_loss_idx]
                selected_loss_rec = full_batch_reconstruction_loss[selected_loss_idx]
                selected_loss_reg = full_batch_regularization_loss[selected_loss_idx]

        ## Run optimizer (backprop)
        self.model_train_var = tf.compat.v1.trainable_variables()
        if self.training and not self.planning:
            deselect_var_name = ["training/model_variables/" + m + "_z_posterior_src_zero:0" for m in self.modalities] # don't train src_zero here
            opt_train_var = [var for var in self.model_train_var if var.name not in deselect_var_name]
        elif self.planning:
            if not self.vb_hybrid_posterior_src:
                if not self.vb_reset_posterior:
                    select_var_name = ["training/model_variables/" + m + "_z_posterior_src:0" for m in self.modalities]
                else:
                    select_var_name = ["training/model_variables/" + m + "_z_posterior_src_zero:0" for m in self.modalities]
            else:
                select_var_name = ["training/model_variables/" + m + "_z_posterior_src:0" for m in self.modalities]
                select_var_name.append(["training/model_variables/" + m + "_z_posterior_src_zero:0" for m in self.modalities])
            if self.vb_posterior_src_extend:
                select_var_name.append([var for var in self.model_train_var if "z_posterior_from_src" in var.name])
            # select_var_name = []
            # for v in self.model_train_var:
            #     select_var_name.append(v.name)
            opt_train_var = [var for var in self.model_train_var if var.name in select_var_name]
        else:
            opt_train_var = None

        print("**Trainable variables in use**")
        vidx = 0
        if opt_train_var is not None:
            for v in opt_train_var:
                print(str(vidx) + " " + str(v.name) + " " + str(v.get_shape()))
                vidx += 1
        else:
            print("None")

        if self.training:
            # Supported optimizers: ADAM (default), Gradient descent, Momentum, Adagrad, RMSProp
            if self.optimizer_func == "gradient_descent":
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer_func == "momentum":
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)
            elif self.optimizer_func == "adagrad":
                optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer_func == "rmsprop":
                optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon=self.optimizer_epsilon)#, decay=0.1)#, momentum=0.5)
            elif self.optimizer_func == "rmsprop_momentum":
                optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon=self.optimizer_epsilon, momentum=0.5, centered=True)
            elif self.optimizer_func == "adam":
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.optimizer_epsilon)
            gradients, variables = list(zip(*optimizer.compute_gradients(total_batch_loss, var_list=opt_train_var)))
            if self.gradient_clip > 0:
                pruned_gradients = [tf.compat.v1.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) if grad is not None else None for grad in gradients]
                pruned_gradients = [tf.compat.v1.where(tf.math.is_inf(grad), tf.constant(self.gradient_clip, shape=np.shape(grad)), grad) if grad is not None else None for grad in pruned_gradients]
                clipped_gradients, _ = tf.clip_by_global_norm(pruned_gradients, self.gradient_clip)
                training_batch = optimizer.apply_gradients(list(zip(clipped_gradients, variables)))
            else:
                training_batch = optimizer.apply_gradients(list(zip(gradients, variables)))
        else:
            optimizer = None
            training_batch = None

        ## Save model summary
        with tf.compat.v1.name_scope("loss"):
            for m in self.modalities:
                rec_loss = batch_reconstruction_loss[m][0]
                tf.compat.v1.summary.scalar(m + "_batch_reconstruction_loss", rec_loss / data["timesteps"])
                if self.vb_meta_prior[m][0] != -1 and self.training:
                    for i in range(1, len(self.z_units[m])):
                        reg_loss = batch_regularization_loss[m][i][0]
                        tf.compat.v1.summary.scalar(m + "_z" + str(i) + "_batch_regularization_loss", -reg_loss / data["timesteps"])
            tf.compat.v1.summary.scalar("total_batch_loss", total_batch_loss / data["timesteps"])

        self.saver = tf.compat.v1.train.Saver(var_list=self.model_train_var, max_to_keep=None)

        ## Output the model
        model = dict()
        model["data"] = data
        model["data_length"] = data["timesteps"]
        model["generated_out"] = {m: [tf.transpose(a=output_model["out"][m][i], perm=[1, 0, 2], name="transpose_generated_out_all") for i in range(len(output_model["out"][m]))] for m in self.modalities}
        model["initial"] = {m: [tf.transpose(a=output_model["out_initial"][m][i], perm=[1, 0, 2], name="transpose_initial") for i in range(len(output_model["out_initial"][m]))] for m in self.modalities}
        model["generated_z_prior_mean"] = generated_z_prior_mean
        model["generated_z_prior_var"] = generated_z_prior_var
        model["generated_z_prior"] = generated_z_prior
        model["generated_z_posterior_mean"] = generated_z_posterior_mean
        model["generated_z_posterior_var"] = generated_z_posterior_var
        model["generated_z_posterior"] = generated_z_posterior
        model["generated_z_posterior_src"] = generated_z_posterior_src
        model["batch_reconstruction_loss"] = {m: batch_reconstruction_loss[m] for m in self.modalities}
        model["batch_regularization_loss"] = {m: batch_regularization_loss[m] for m in self.modalities}
        model["total_batch_reconstruction_loss"] = total_batch_reconstruction_loss
        model["total_batch_regularization_loss"] = total_batch_regularization_loss
        model["total_batch_loss"] = total_batch_loss
        model["initial_states"] = initial_states
        model["training_batch"] = training_batch
        # if self.training:
        #     model["gradients"] = {variables[i].name: gradients[i].values for i in xrange(len(gradients))}
        if self.planning:
            model["selected_loss_idx"] = selected_loss_idx
            model["selected_loss"] = selected_loss
            model["selected_loss_rec"] = selected_loss_rec
            model["selected_loss_reg"] = selected_loss_reg
    
        return model
    # build_model #

    def build_model_one_step_scan(self, previous_states, current_input):
        with tf.compat.v1.variable_scope('model_variables', reuse=tf.compat.v1.AUTO_REUSE):
            return self.build_model_one_step(previous_states, current_input)

    def calculate_z_prior(self, idx_layer, out, modality, scope=None, override_l=None, override_sigma=None, override_myu=None, override_epsilon=None):
        scope_name = 'l' + str(idx_layer) + '_' + modality
        if scope is not None:
            scope_name += '_' + str(scope)
        if override_l is not None and (override_l == True or idx_layer in override_l):
            return ops.calculate_z_prior(idx_layer, out, self.d_neurons[modality], self.z_units[modality], self.batch_size, self.z_activation_func[modality], scope_name, override_sigma=override_sigma, override_myu=override_myu, override_epsilon=override_epsilon)
        else:
            return ops.calculate_z_prior(idx_layer, out, self.d_neurons[modality], self.z_units[modality], self.batch_size, self.z_activation_func[modality], scope_name, limit_sigma=self.vb_limit_sigma)

    def calculate_z_posterior(self, idx_layer, out, source, modality, scope=None, override_l=None, override_sigma=None, override_myu=None, override_epsilon=None):
        scope_name = 'l' + str(idx_layer) + '_' + modality
        if scope is not None:
            scope_name += '_' + str(scope)
        
        if self.vb_posterior_past_input:
            posterior_in = out
        else:
            posterior_in = None

        if override_l is not None and (override_l == True or idx_layer in override_l):
            return ops.calculate_z_posterior(idx_layer, posterior_in, source, self.d_neurons[modality], self.z_units[modality], self.batch_size, self.z_activation_func[modality], scope_name, override_sigma=override_sigma, override_myu=override_myu, override_epsilon=override_epsilon, source_extend=self.vb_posterior_src_extend)
        else:
            return ops.calculate_z_posterior(idx_layer, posterior_in, source, self.d_neurons[modality], self.z_units[modality], self.batch_size, self.z_activation_func[modality], scope_name, limit_sigma=self.vb_limit_sigma, source_extend=self.vb_posterior_src_extend)

    def build_model_one_step(self, previous_states, current_input):
        if self.vb_hybrid_posterior_src_idx_override is None:
            idx_seq = current_input[0]
        else:
            idx_seq = tf.constant(self.vb_hybrid_posterior_src_idx_override)
        t_step = previous_states["t_step"]

        out_initial = previous_states["out_initial"]
        out = {m: self.n_layers[m]*[None] for m in self.modalities}
        state = {m: self.n_layers[m]*[None] for m in self.modalities}
        z_prior_mean = {m: self.n_layers[m]*[None] for m in self.modalities}
        z_prior_var = {m: self.n_layers[m]*[None] for m in self.modalities}
        z_prior = {m: self.n_layers[m]*[None] for m in self.modalities}
        z_posterior_mean = {m: self.n_layers[m]*[None] for m in self.modalities}
        z_posterior_var = {m: self.n_layers[m]*[None] for m in self.modalities}
        z_posterior = {m: self.n_layers[m]*[None] for m in self.modalities}

        if self.override_d_output is not None:
            for m in self.modalities:
                for i in range(max(self.n_layers.values())-1, 0, -1):
                    fixed_d = tf.reshape(tf.tile(tf.gather(tf.convert_to_tensor(value=self.override_d_output[m][i], name="load_override_d"), t_step), [self.n_seq]), [self.n_seq, self.d_neurons[m][i]])
                    previous_states["out"][m][i] = tf.compat.v1.where(tf.logical_and(tf.greater_equal(t_step, self.override_d_output_range[0]), tf.less(t_step, self.override_d_output_range[1])), fixed_d, previous_states["out"][m][i], name="where_override_d_range")

        if not self.vb_zero_initial_out:
            for m in self.modalities:
                out_initial_var = tf.compat.v1.get_variable(m + "_out_initial", shape=[1], initializer=tf.compat.v1.zeros_initializer, trainable=self.training) # TODO: reuse for continued generation?
                previous_states["out"][m] = tf.compat.v1.where(tf.equal(t_step, 0), out_initial_var, previous_states["out"][m], name="where_initial_d_check")

        # Layers 1+
        for i in range(max(self.n_layers.values())-1, 0, -1):
            current_z_logits = {m: None for m in self.modalities}
            higher_level_out_logits = {m: None for m in self.modalities}
            lower_level_out_logits = {m: None for m in self.modalities}
            current_level_out_logits = {m: None for m in self.modalities}

            for mi, m in enumerate(self.modalities):
                if i > self.n_layers[m]-1:
                    continue
                lower_level_out = None
                higher_level_out = None
                
                ## Collect previous timestep outputs
                # Input from lower levels
                if i != 1 and self.connect_bottomup_d:
                    # ll input from this modality
                    lower_level_out = previous_states["out"][m][i-1]
                    # If this is a shared layer, gather ll input from all modalities
                    if i == self.n_layers[m]-1 and self.shared_layer == m:
                        for x in self.modalities:
                            if x == m:
                                continue
                            else:
                                lower_level_xout = previous_states["out"][x][i-1]
                                if self.layers_concat_input:
                                    lower_level_out = tf.concat([lower_level_out, lower_level_xout], axis=1, name="concat_ll_out")
                                else:
                                    lower_level_out = tf.add_n([lower_level_out, lower_level_xout], name="addn_ll_out")
                else: # lowest level
                    if self.input_provide_data:
                        lower_level_out = tf.add(self.input_cl_ratio * previous_states["out"][m][0], (1.0 - self.input_cl_ratio) * current_input[mi+1], name="add_ll_inmix") # Mix input and previous output
                    # else nothing enters the lowest level
                
                # Input from higher levels
                if i < self.n_layers[m]-1 and self.connect_topdown_d:
                    # hl input from this modality
                    if not self.connect_topdown_dt:
                        higher_level_out = previous_states["out"][m][i+1]
                    else:
                        higher_level_out = out[m][i+1]
                    if i == max(self.n_layers.values())-2 and self.shared_layer is not None and m != self.shared_layer: # top layer-1
                        if not self.connect_topdown_dt:
                            higher_level_xout = previous_states["out"][self.shared_layer][i+1]
                        else:
                            higher_level_xout = out[self.shared_layer][i+1]
                        if self.layers_concat_input:
                            higher_level_out = tf.concat([higher_level_out, higher_level_xout], axis=1, name="concat_hl_out")
                        else:
                            higher_level_out = tf.add_n([higher_level_out, higher_level_xout], name="addn_hl_out")

                # Input from current level (previous timestep)
                current_level_out = previous_states["out"][m][i]

                with tf.compat.v1.variable_scope('l' + str(i) + '_' + m):
                    lower_level_out_logits[m] = _linear([lower_level_out], self.d_neurons[m][i], bias=True, scope_here=m+"_ll_to_cell") if lower_level_out is not None else tf.zeros([self.batch_size, self.d_neurons[m][i]])
                    higher_level_out_logits[m] = _linear([higher_level_out], self.d_neurons[m][i], bias=True, scope_here=m+"_hl_to_cell") if higher_level_out is not None else tf.zeros([self.batch_size, self.d_neurons[m][i]])
                    current_level_out_logits[m] = _linear([current_level_out], self.d_neurons[m][i], bias=True, scope_here=m+"_cl_to_cell")

                if self.connect_topdown_dz and i < self.n_layers[m]-1:
                    # Independent of connect_d
                    if not self.connect_topdown_dt:
                        d_to_z = previous_states["out"][m][i+1]
                    else:
                        d_to_z = out[m][i+1]
                    if i == max(self.n_layers.values())-2 and self.shared_layer is not None and m != self.shared_layer: # top layer-1
                        if not self.connect_topdown_dt:
                            higher_level_xout = previous_states["out"][self.shared_layer][i+1]
                        else:
                            higher_level_xout = out[self.shared_layer][i+1]
                        if self.layers_concat_input:
                            d_to_z = tf.concat([higher_level_out, higher_level_xout], axis=1, name="concat_dtoz")
                        else:
                            d_to_z = tf.add_n([higher_level_out, higher_level_xout], name="addn_dtoz")
                else:
                    d_to_z = current_level_out

                if self.vb_meta_prior[m][i-1] >= 0 and self.connect_z:
                    ## Calculate prior
                    if self.vb_ugaussian_t_range is None:
                        if self.vb_prior_override_t_range is None:
                            current_z_prior, current_z_prior_mean, current_z_prior_var = self.calculate_z_prior(i, d_to_z, m, override_l=self.vb_prior_override_l, override_sigma=self.vb_prior_override_sigma, override_myu=self.vb_prior_override_myu, override_epsilon=self.vb_prior_override_epsilon)
                        else:
                            current_z_prior, current_z_prior_mean, current_z_prior_var = tf.cond(pred=tf.logical_and(tf.greater_equal(t_step, self.vb_prior_override_t_range[0]), tf.less(t_step, self.vb_prior_override_t_range[1])), true_fn=lambda: self.calculate_z_prior(i, d_to_z, m, override_l=self.vb_prior_override_l, override_sigma=self.vb_prior_override_sigma, override_myu=self.vb_prior_override_myu, override_epsilon=self.vb_prior_override_epsilon), false_fn=lambda: self.calculate_z_prior(i, d_to_z, m), name="cond_z_p_range_t")
                    else:
                        if self.vb_prior_override_t_range is None:
                            current_z_prior, current_z_prior_mean, current_z_prior_var = tf.cond(pred=tf.logical_and(tf.greater_equal(t_step, self.vb_ugaussian_t_range[0]), tf.less(t_step, self.vb_ugaussian_t_range[1])), true_fn=lambda: self.calculate_z_prior(i, d_to_z, m, override_l=True, override_sigma=1.0, override_myu=0.0), false_fn=lambda: self.calculate_z_prior(i, d_to_z, m, override_l=self.vb_prior_override_l, override_sigma=self.vb_prior_override_sigma, override_myu=self.vb_prior_override_myu, override_epsilon=self.vb_prior_override_epsilon), name="cond_z_p_range_u")
                        else:
                            current_z_prior, current_z_prior_mean, current_z_prior_var = tf.cond(pred=tf.logical_and(tf.greater_equal(t_step, self.vb_ugaussian_t_range[0]), tf.less(t_step, self.vb_ugaussian_t_range[1])), true_fn=lambda: self.calculate_z_prior(i, d_to_z, m, override_l=True, override_sigma=1.0, override_myu=0.0), false_fn=lambda: tf.cond(pred=tf.logical_and(tf.greater_equal(t_step, self.vb_prior_override_t_range[0]), tf.less(t_step, self.vb_prior_override_t_range[1])), true_fn=lambda: self.calculate_z_prior(i, d_to_z, m, override_l=self.vb_prior_override_l, override_sigma=self.vb_prior_override_sigma, override_myu=self.vb_prior_override_myu, override_epsilon=self.vb_prior_override_epsilon), false_fn=lambda: self.calculate_z_prior(i, d_to_z, m)), name="cond_z_p_range_tu")
                    if self.vb_seq_prior[m][i]:
                        ## Calculate posterior
                        # Load the correct posterior source
                        src_z = max(self.z_units[m]) if self.vb_posterior_src_extend else max(self.z_units[m])*2
                        if not self.vb_hybrid_posterior_src:
                            _ = tf.compat.v1.get_variable(m + "_z_posterior_src_zero", shape=[self.max_timesteps, self.n_layers[m], self.n_seq, src_z], initializer=tf.compat.v1.zeros_initializer, trainable=True) # this src is reserved in case we don't want to use the primary
                            if not self.vb_reset_posterior:
                                z_posterior_src_var_name = m + "_z_posterior_src"
                            else:
                                z_posterior_src_var_name = m + "_z_posterior_src_zero"
                            full_z_posterior_src = tf.compat.v1.get_variable(z_posterior_src_var_name, shape=[self.max_timesteps, self.n_layers[m], self.n_seq, src_z], initializer=tf.compat.v1.zeros_initializer, trainable=True)
                        else:
                            initial_start = self.vb_hybrid_posterior_src_range[0]
                            initial_end = self.vb_hybrid_posterior_src_range[1]
                            alt_z_posterior_src = tf.compat.v1.get_variable(m + "_z_posterior_src_zero", shape=[self.max_timesteps, self.n_layers[m], self.n_seq, src_z], initializer=tf.compat.v1.zeros_initializer, trainable=True)
                            z_posterior_src = tf.compat.v1.get_variable(m + "_z_posterior_src", shape=[self.max_timesteps, self.n_layers[m], self.n_seq, src_z], initializer=tf.compat.v1.zeros_initializer, trainable=True)
                            if not self.vb_hybrid_posterior_src_zero_init:
                                full_z_posterior_src = tf.concat([z_posterior_src[initial_start:initial_end, :, :, :], alt_z_posterior_src[initial_end:, :, :, :]], axis=0, name="concat_z_qsrc_hybrid0")
                            else:
                                full_z_posterior_src = tf.concat([alt_z_posterior_src[initial_start:initial_end, :, :, :], z_posterior_src[initial_end:, :, :, :]], axis=0, name="concat_z_qsrc_hybrida")
                        z_posterior_src = tf.gather(full_z_posterior_src, t_step, name="gather_z_qsrc_t")
                        current_z_posterior_src = tf.gather(z_posterior_src[i], idx_seq, name="gather_z_qsrc_idx") # reorder posterior d of this layer to match data sequences
                        if self.vb_posterior_override_t_range is None:
                            current_z_posterior, current_z_posterior_mean, current_z_posterior_var = self.calculate_z_posterior(i, d_to_z, current_z_posterior_src, m, override_l=self.vb_posterior_override_l, override_sigma=self.vb_posterior_override_sigma, override_myu=self.vb_posterior_override_myu, override_epsilon=self.vb_posterior_override_epsilon)
                        else:
                            current_z_posterior, current_z_posterior_mean, current_z_posterior_var = tf.cond(pred=tf.logical_and(tf.greater_equal(t_step, self.vb_posterior_override_t_range[0]), tf.less(t_step, self.vb_posterior_override_t_range[1])), true_fn=lambda: self.calculate_z_posterior(i, d_to_z, current_z_posterior_src, m, override_l=self.vb_posterior_override_l, override_sigma=self.vb_posterior_override_sigma, override_myu=self.vb_posterior_override_myu, override_epsilon=self.vb_posterior_override_epsilon), false_fn=lambda: self.calculate_z_posterior(i, d_to_z, current_z_posterior_src, m), name="cond_z_q_range")
                    else:
                        current_z_posterior = None
                        current_z_posterior_mean = previous_states["z_posterior_mean"][m][i]
                        current_z_posterior_var = previous_states["z_posterior_var"][m][i]
                else: # No-op
                    current_z_prior = None
                    current_z_prior_mean = previous_states["z_prior_mean"][m][i]
                    current_z_prior_var = previous_states["z_prior_var"][m][i]
                    current_z_posterior = None
                    current_z_posterior_mean = previous_states["z_posterior_mean"][m][i]
                    current_z_posterior_var = previous_states["z_posterior_var"][m][i]

                # Select current Z
                if self.vb_seq_prior[m][i]:
                    if self.vb_posterior_blend_factor > 0.0:
                        current_z = tf.add_n([tf.multiply(current_z_prior, 1-self.vb_posterior_blend_factor), tf.multiply(current_z_posterior, self.vb_posterior_blend_factor)], name="addn_z_pq_blend")
                    elif self.vb_hybrid_posterior_src:
                        initial_start = self.vb_hybrid_posterior_src_range[0]
                        initial_end = self.vb_hybrid_posterior_src_range[1]
                        if self.vb_hybrid_prior_override:
                            current_z_prior = tf.compat.v1.where(tf.logical_and(tf.greater_equal(t_step, initial_start), tf.less(t_step, initial_end)), current_z_posterior, current_z_prior, name="where_z_p_range")
                            current_z_prior_mean = tf.compat.v1.where(tf.logical_and(tf.greater_equal(t_step, initial_start), tf.less(t_step, initial_end)), current_z_posterior_mean, current_z_prior_mean, name="where_zm_p_range")
                            current_z_prior_var = tf.compat.v1.where(tf.logical_and(tf.greater_equal(t_step, initial_start), tf.less(t_step, initial_end)), current_z_posterior_var, current_z_prior_var, name="where_zv_p_range")
                        current_z = tf.compat.v1.where(tf.logical_and(tf.greater_equal(t_step, initial_start), tf.less(t_step, initial_end)), current_z_posterior, current_z_prior, name="where_z_pq_range")
                    else:
                        current_z = current_z_posterior if not self.vb_prior_output else current_z_prior
                else:
                    current_z = current_z_prior

                z_prior[m][i] = current_z_prior if current_z_prior is not None else previous_states["z_prior"][m][i]
                z_prior_mean[m][i] = current_z_prior_mean
                z_prior_var[m][i] = current_z_prior_var
                z_posterior[m][i] = current_z_posterior if current_z_posterior is not None else previous_states["z_posterior"][m][i]
                z_posterior_mean[m][i] = current_z_posterior_mean
                z_posterior_var[m][i] = current_z_posterior_var

                with tf.compat.v1.variable_scope('l' + str(i) + '_' + m):
                    current_z_logits[m] = _linear([current_z], self.d_neurons[m][i], bias=True, scope_here=m+"_z_to_cell") if current_z is not None else tf.zeros([self.batch_size, self.d_neurons[m][i]])

            ## Synchronize level

            for m in self.modalities:
                if i > self.n_layers[m]-1:
                    continue
                # Add horizontal and vertical connections in this layer
                if self.gradient_clip_input == -1: # special case: layer normalization
                    # L2 norm on d and z separately
                    z_logits_norm = tf.nn.l2_normalize(current_z_logits[m], axis=1, name="normalize_z")
                    if self.layers_concat_input:
                        d_logits = tf.concat([lower_level_out_logits[m], higher_level_out_logits[m], current_level_out_logits[m]], axis=1, name="concat_d")
                    else:
                        d_logits = tf.add_n([lower_level_out_logits[m], higher_level_out_logits[m], current_level_out_logits[m]], name="addn_d")
                    d_logits_norm = tf.nn.l2_normalize(d_logits, axis=1)
                    level_output = [z_logits_norm, d_logits_norm]
                else:
                    level_output = [current_z_logits[m], lower_level_out_logits[m], higher_level_out_logits[m], current_level_out_logits[m]]

                if self.connect_horizontal:
                    for x in self.modalities:
                        if x == m:
                            continue
                        if current_level_out_logits[x] is not None:
                            level_output.append(current_level_out[x]) # current level output from all modalities

                if self.layers_concat_input:
                    sum_level_output = tf.concat(level_output, axis=1, name="concat_l_out")
                else:
                    sum_level_output = tf.add_n(level_output, name="addn_l_out")

                # There's no gradient here, but keeping it for bc
                if self.gradient_clip_input > 0: # clip input
                    sum_level_output = tf.clip_by_norm(sum_level_output, self.gradient_clip_input, name="clip_l_out")

                # Finally compute D
                out[m][i], state[m][i], _ = self.layers[m][i](sum_level_output, previous_states["state"][m][i], scope=self.layers_names[m][i]) # TODO: read internal (gate) states?

        ## Synchronize modalities

        for m in self.modalities:
            # Layer 0 a.k.a. output layer
            with tf.compat.v1.variable_scope('l0_' + m):
                # Layer 0 has no Z units
                z_prior[m][0] = previous_states["z_prior"][m][0]
                z_prior_mean[m][0] = previous_states["z_prior_mean"][m][0]
                z_prior_var[m][0] = previous_states["z_prior_var"][m][0]
                z_posterior[m][0] = previous_states["z_posterior"][m][0]
                z_posterior_mean[m][0] = previous_states["z_posterior_mean"][m][0]
                z_posterior_var[m][0] = previous_states["z_posterior_var"][m][0]

                # Final output
                l0_o = _linear(out[m][1], self.dims[m] * self.softmax_quant, bias=True, scope_here=m+"_to_out")
                if self.output_z_factor[m] > 0.0:
                    z_to_output = z_prior[m][1] if self.vb_prior_output else z_posterior[m][1]
                    l1_z_logits = _linear(z_to_output, self.dims[m] * self.softmax_quant, bias=True, scope_here=m+"_z_blend_output") if z_to_output is not None else tf.zeros([self.batch_size, self.dims[m] * self.softmax_quant])
                    l0_o += self.output_z_factor[m] * l1_z_logits
                l0_softmax = []
                for i in range(self.dims[m]):
                    l0_softmax.append(tf.nn.softmax(l0_o[:, self.softmax_quant*i:self.softmax_quant*(i+1)], name="softmax_l0_output"))
                out[m][0] = tf.concat(l0_softmax, 1, name="concat_l0_output")
                state[m][0] = l0_o

        return ops.internal_states_dict(t_step=t_step+1, out=out, out_initial=out_initial, state=state, 
                                        z_prior_mean=z_prior_mean, z_prior_var=z_prior_var, z_prior=z_prior, 
                                        z_posterior_mean=z_posterior_mean, z_posterior_var=z_posterior_var, z_posterior=z_posterior)

                                        
    ## Output parts of the model to human readable format
    # fig_idx: True = plot all outputs in one figure, False = disabled, <integer> = plot specified output only, None = also save single figure for each output
    # fig_xy: True = save lissajous plot of first two dimensions, False = disabled
    # NB: motor_output can start at t=0 and have effectively sequence length+1 steps. Other outputs start at t=1
    def write_file_csv(self, generated, epoch, modality, idx=None, layer=0, filename_prefix=None, dir=None, fig_idx=True, fig_plot_markers=False, initial=None, fig_plot_dims=None, compute_entropy=False, override_d=None):
        # generated_all_layers = np.squeeze(np.asarray(generated), axis=0)
        generated_all_layers = generated[0]

        # Select which layer to save
        if layer is None:
            min_layer = 1
            max_layer = self.n_layers[modality] # output for all layers (except final)
        else:
            min_layer = layer
            max_layer = layer+1

        for l in range(min_layer, max_layer):
            if idx is None:
                min_seq = 0
                max_seq = self.n_seq # output for each training sequence
            else:
                min_seq = idx
                max_seq = idx+1

            lgenerated = np.asarray(generated_all_layers[l])
            if compute_entropy and lgenerated.shape[2] > 1:
                Hd = []
                for n in range(lgenerated.shape[0]):
                    Hd.append([entropy(lgenerated[n, step, :]**2+ops.eps_minval) for step in range(lgenerated.shape[1])])
                lgenerated = np.array(Hd)
                lgenerated = np.expand_dims(lgenerated, axis=2)
            elif override_d is not None:
                source_d = np.delete(override_d[modality][l], 0, axis=0) # drop t=0
                delta_d = lgenerated - source_d
                mean_d = np.mean(delta_d, axis=0)
                rms_d = np.sqrt(np.mean(np.square(delta_d-mean_d), axis=0))
                rms_v = np.sqrt(np.mean(np.linalg.norm(delta_d-mean_d, axis=2), axis=0))
                lgenerated = np.expand_dims(rms_d, axis=0)
                min_seq = 0
                max_seq = 1

            for n in range(min_seq, max_seq):
                if filename_prefix:
                    filename = str(filename_prefix) + '_' + modality + "_d"  + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.csv"
                else:
                    filename = modality + "_d"  + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.csv"

                if dir is not None:
                    filename = dir + '/' + filename
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                
                file_write = open(filename, 'w')

                ## Initial row
                # Columns for CSV
                for x in range(lgenerated.shape[2]): # output dims
                    if x == 0:
                        file_write.write("d%d_%d" % (l, x))
                    else:
                        file_write.write(",d%d_%d" % (l, x))
                file_write.write("\n")

                if initial is not None: # print d0
                    initial_vals = np.asarray(initial[0][l])
                    for x in range(lgenerated.shape[2]): # output dims
                        if x == 0:
                            file_write.write("%f" % initial_vals[n, 0, x])
                        else:
                            file_write.write(",%f" % initial_vals[n, 0, x])
                    file_write.write("\n")
                else: # don't print d0
                    for x in range(lgenerated.shape[2]): # output dims
                        file_write.write(",")
                    file_write.write("\n")

                for step in range(lgenerated.shape[1]): # total timesteps
                    for x in range(lgenerated.shape[2]): # output dims
                        if x == 0:
                            file_write.write("%f" % lgenerated[n, step, x])    
                        else:
                           file_write.write(",%f" % lgenerated[n, step, x])
                    file_write.write("\n")
                file_write.close()

            if fig_idx is not False:
                # Select what and how to plot
                if fig_idx is None:
                    min_seq = 0
                    max_seq = lgenerated.shape[0] # output for each training sequence
                    save_after_each = True
                elif fig_idx is not True:
                    min_seq = fig_idx
                    max_seq = fig_idx+1 # save a specific output sequence
                    save_after_each = True
                else:
                    min_seq = 0
                    max_seq = lgenerated.shape[0]
                    save_after_each = False # stack all outputs in one plot

                fig, ax = plt.subplots()
                fig.tight_layout()

                for n in range(min_seq, max_seq):
                    if save_after_each:
                        fig_i, ax_i = plt.subplots()
                        fig_i.tight_layout()

                    for x in range(lgenerated.shape[2]): # for each dimension
                        if fig_plot_dims is not None and x not in fig_plot_dims:
                            continue
                        if initial is not None:
                            initial_vals = np.asarray(initial[0][l])
                            x_plot = np.linspace(0, lgenerated.shape[1]+1, lgenerated.shape[1]+1) # timesteps on X
                            y_plot = np.concatenate((np.expand_dims(initial_vals[n, 0, x], axis=0), lgenerated[n, :, x]))
                        else:
                            x_plot = np.linspace(1, lgenerated.shape[1]+1, lgenerated.shape[1]) # timesteps on X
                            y_plot = lgenerated[n, :, x]

                        if fig_plot_markers:
                            ax.scatter(x_plot, y_plot)
                        ax.plot(x_plot, y_plot)
                        ax.set_xlim(left=0.0)

                        if save_after_each:
                            if fig_plot_markers:
                                ax_i.scatter(x_plot, y_plot)
                            ax_i.plot(x_plot, y_plot)
                            ax_i.set_xlim(left=0.0)

                    if save_after_each:
                        if filename_prefix:
                            figfilename = str(filename_prefix) + '_' + modality + "_d"  + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.png"
                        else:
                            figfilename = modality + "_d"  + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.png"
                        if dir is not None:
                            figfilename = dir + '/' + figfilename
                            if not os.path.exists(dir):
                                os.makedirs(dir)

                        fig_i.savefig(figfilename)
                        plt.close(fig_i)

                if filename_prefix:
                    figfilename = str(filename_prefix) + '_' + modality + "_d"  + str(l) + "_e" + str(epoch) + "_out.png"
                else:
                    figfilename = modality + "_d"  + str(l) + "_e" + str(epoch) + "_out.png"
                if dir is not None:
                    figfilename = dir + '/' + figfilename
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                if override_d is not None: # replot stacked
                    if initial is not None:
                        initial_vals = np.asarray(initial[0][l])
                        y_plot = np.concatenate((np.expand_dims(initial_vals[0, 0, 0], axis=0), rms_v))
                    else:
                        y_plot = np.squeeze(lgenerated[:, :, :], axis=2)
                    ax.clear()
                    if fig_plot_markers:
                        ax.scatter(x_plot, y_plot)
                    ax.plot(x_plot, y_plot)
                    ax.set_xlim(left=0.0)

                fig.savefig(figfilename)
                plt.close(fig)

    # Softmax decoded
    def write_file_decoded(self, generated, epoch, modality, idx=None, filename_prefix=None, dir=None, sm_min=-1, sm_max=1, fig_idx=True, fig_xy=False, fig_plot_markers=False, scale=None, fig_plot_dims=None, fig_x_lim=[-0.5, 0.5], fig_y_lim=[0.0, 1.0], compute_entropy=False, override_d=None, skip_csv=False):
        generated = generated[0][0] # Decode only layer 0 output
        decoded_all = np.zeros((self.n_seq, self.max_timesteps, self.dims[modality]))

        if idx is None:
            min_seq = 0
            max_seq = self.n_seq # Output for each training sequence
        else:
            min_seq = idx
            max_seq = idx+1

        if scale is not None:
            scaler_min_range = np.load(scale)

        # Decode first
        for n in range(min_seq, max_seq):
            if self.softmax_quant > 1:
                decoded_all[n, :, :] = ops.unsoftmax(generated[n, :, :], sm_minVal=sm_min, sm_maxVal=sm_max, softmax_quant=self.softmax_quant)
            else:
                decoded_all[n, :, :] = generated[n, :, :]

            if scale is not None:
                decoded_all[n, :, :] = (decoded_all[n, :, :] * scaler_min_range[n, self.dims[modality]:self.dims[modality]*2]) + scaler_min_range[n, 0:self.dims[modality]]

            if compute_entropy and override_d is None and decoded_all.shape[2] > 1:
                decoded_all[n, :, :] = np.expand_dims(np.array([entropy(decoded_all[n, step, :]**2+ops.eps_minval) for step in range(decoded_all.shape[1])]), axis=2)

        if override_d is not None:
            source_d = override_d[modality][0]
            delta_d = decoded_all - source_d
            mean_d = np.mean(delta_d, axis=0)
            rms_d = np.sqrt(np.mean(np.square(delta_d-mean_d), axis=0))
            rms_v = np.sqrt(np.mean(np.linalg.norm(delta_d-mean_d, axis=2), axis=0))
            decoded_all = np.expand_dims(rms_d, axis=0)
            min_seq = 0
            max_seq = 1

        if not skip_csv:
            for n in range(min_seq, max_seq):
                decoded = decoded_all[n, :, :]
                if filename_prefix:
                    filename = str(filename_prefix) + '_' + modality + "_d0_e" + str(epoch) + "_n" + str(n) + "_decoded.csv"
                else:
                    filename = modality + "_d0_e" + str(epoch) + "_n" + str(n) + "_decoded.csv"

                if dir is not None:
                    filename = dir + '/' + filename
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                
                file_write = open(filename, 'w')

                ## Initial row
                # Columns for CSV
                for x in range(decoded.shape[1]): # output dims
                    if x == 0:
                        file_write.write("J%d" % x)
                    else:
                        file_write.write(",J%d" % x)
                file_write.write("\n")
                # Skip t=0
                for x in range(decoded.shape[1]): # output dims
                    file_write.write(",")
                file_write.write("\n")

                for step in range(generated.shape[1]): # total timesteps
                    for x in range(decoded.shape[1]): # output dims
                        if x == 0:
                            file_write.write("%f" % decoded[step, x])
                        else:
                            file_write.write(",%f" % decoded[step, x])
                    file_write.write("\n")
                file_write.close()

        save_stacked = False
        if fig_idx is not False:
            # Select what and how to plot
            if fig_idx is None:
                min_seq = 0
                max_seq = decoded_all.shape[0] # Output for each training sequence
                save_after_each = True
                save_stacked = True
            elif fig_idx is not True:
                min_seq = fig_idx
                max_seq = fig_idx+1 # Save a specific output sequence
                save_after_each = True
                save_stacked = False
            else:
                min_seq = 0
                max_seq = decoded_all.shape[0]
                save_after_each = False # Stack all outputs in one plot
                save_stacked = True

            if save_stacked:
                fig1, ax1 = plt.subplots()
                fig1.tight_layout()

            if fig_xy:
                fig2, ax2 = plt.subplots()
                fig2.tight_layout()
                ax2.set_xlim((fig_x_lim[0], fig_x_lim[1]))
                ax2.set_ylim((fig_y_lim[0], fig_y_lim[1]))

            for n in range(min_seq, max_seq):
                if save_after_each:
                    fig_i, ax_i = plt.subplots()
                    fig_i.tight_layout()
                
                decoded = decoded_all[n, :, :]

                for x in range(decoded.shape[1]): # for each dimension
                    if fig_plot_dims is not None and x not in fig_plot_dims:
                        continue
                    x_plot = np.linspace(1, generated.shape[1]+1, generated.shape[1]) # timesteps on X
                    y_plot = decoded[:, x]

                    if save_stacked:
                        if fig_plot_markers:
                            ax1.scatter(x_plot, y_plot)
                        ax1.plot(x_plot, y_plot)
                        ax1.set_xlim(left=0.0)

                    if save_after_each:
                        if fig_plot_markers:
                            ax_i.scatter(x_plot, y_plot)
                        ax_i.plot(x_plot, y_plot)
                        ax_i.set_xlim(left=0.0)

                if save_after_each:
                    if filename_prefix:
                        figfilename = str(filename_prefix) + '_' + modality + "_d0_e" + str(epoch) + "_n" + str(n) + "_decoded.png"
                    else:
                        figfilename = modality + "_d0_e" + str(epoch) + "_n" + str(n) + "_decoded.png"
                    if dir is not None:
                        figfilename = dir + '/' + figfilename
                        if not os.path.exists(dir):
                            os.makedirs(dir)

                    fig_i.savefig(figfilename)
                    plt.close(fig_i)

                if fig_xy:
                    if save_after_each:
                        fig_i, ax_i = plt.subplots()
                        fig_i.tight_layout()
                        ax_i.set_xlim((fig_x_lim[0], fig_x_lim[1]))
                        ax_i.set_ylim((fig_y_lim[0], fig_y_lim[1]))

                    if fig_plot_dims is None:
                        x_plot = decoded[:, 0] # X is dim 0
                        y_plot = decoded[:, 1] # Y is dim 1
                    else:
                        x_plot = decoded[:, fig_plot_dims[0]] # X is at dim 0
                        y_plot = decoded[:, fig_plot_dims[1]] # Y is at dim 1

                    if fig_plot_markers:
                        ax2.scatter(x_plot, y_plot)
                    ax2.plot(x_plot, y_plot)

                    if save_after_each:
                        if fig_plot_markers:
                            ax_i.scatter(x_plot, y_plot)
                        ax_i.plot(x_plot, y_plot)

                        if filename_prefix:
                            figfilename = str(filename_prefix) + '_' + modality + "_d0_e" + str(epoch) + "_n" + str(n) + "_decoded_xy.png"                    
                        else:
                            figfilename = modality + "_d0_e" + str(epoch) + "_n" + str(n) + "_decoded_xy.png"
                        if dir is not None:
                            figfilename = dir + '/' + figfilename
                            if not os.path.exists(dir):
                                os.makedirs(dir)

                        fig_i.savefig(figfilename)
                        plt.close(fig_i)

            if save_stacked:
                if filename_prefix:
                    figfilename = str(filename_prefix) + '_' + modality + "_d0_e" + str(epoch) + "_decoded.png"   
                else:
                    figfilename = modality + "_d0_e" + str(epoch) + "_decoded.png"
                if dir is not None:
                    figfilename = dir + '/' + figfilename
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                if override_d is not None: # replot stacked
                    y_plot = rms_v
                    ax1.clear()
                    if fig_plot_markers:
                        ax1.scatter(x_plot, y_plot)
                    ax1.plot(x_plot, y_plot)
                    ax1.set_xlim(left=0.0)
                fig1.savefig(figfilename)
                plt.close(fig1)

                if fig_xy:
                    if filename_prefix:
                        figfilename = str(filename_prefix) + '_' + modality + "_d0_e" + str(epoch) + "_decoded_xy.png"
                    else:
                        figfilename = modality + "_d0_e" + str(epoch) + "_decoded_xy.png"
                    if dir is not None:
                        figfilename = dir + '/' + figfilename
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                    fig2.savefig(figfilename)
                    plt.close(fig2)


    def write_file_z(self, generated, epoch, modality, idx, layer=None, filename_prefix=None, dir=None, fig_idx=True, fig_ymin=-1.0, fig_ymax=1.0, fig_plot_markers=False, compute_entropy=False, compute_mean=False, compute_mad=False):
        generated = generated[0]
        # Select which layer to save
        if layer is None:
            min_layer = 1
            max_layer = self.n_layers[modality] # Output for each training sequence
        else:
            min_layer = layer
            max_layer = layer+1

        if not isinstance(fig_ymin, (list, tuple, np.ndarray)):
            fig_ymin = np.repeat(fig_ymin, max_layer-min_layer)

        if not isinstance(fig_ymax, (list, tuple, np.ndarray)):
            fig_ymax = np.repeat(fig_ymax, max_layer-min_layer)

        for l in range(min_layer, max_layer):
            layer_z = np.asarray(generated[l])
            if compute_entropy and layer_z.shape[2] > 1:
                Hz = []
                for n in range(layer_z.shape[0]):
                    Hz.append([entropy(layer_z[n, step, :]**2+ops.eps_minval) for step in range(layer_z.shape[1])])
                layer_z = np.array(Hz)
                layer_z = np.expand_dims(layer_z, axis=2)

            # Select which sequence to save
            if idx is None:
                min_seq = 0
                max_seq = self.n_seq # Output for each training sequence
            else:
                min_seq = idx
                max_seq = idx+1
            for n in range(min_seq, max_seq):
                if filename_prefix:
                    filename_z = str(filename_prefix) + '_' + modality + "_z" + str(l) + "_e" + str(epoch) + "_n" + str(n) + ".csv"
                else:
                    filename_z = modality + "_z" + str(l) + "_e" + str(epoch) + "_n" + str(n) + ".csv"

                if dir is not None:
                    filename_z = dir + '/' + filename_z
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                
                file_write_z = open(filename_z, 'w')

                ## Initial row
                # Columns for CSV
                for x in range(layer_z.shape[2]): # output dims
                    if x == 0:
                        file_write_z.write("z%d_%d" % (l, x))
                    else:
                        file_write_z.write(",z%d_%d" % (l, x))
                file_write_z.write("\n")
                # Skip t=0
                for x in range(layer_z.shape[2]): # output dims
                    file_write_z.write(",")
                file_write_z.write("\n")

                for step in range(self.max_timesteps): # total timesteps
                    for x in range(layer_z.shape[2]): # units
                        if x == 0:
                            file_write_z.write("%f" % layer_z[n, step, x])
                        else:
                            file_write_z.write(",%f" % layer_z[n, step, x])
                    file_write_z.write("\n")
                file_write_z.close()

            save_stacked = False
            if fig_idx is not False:
                # Select what and how to plot
                if fig_idx is None:
                    min_seq = 0
                    max_seq = self.n_seq # Output for each training sequence
                    save_after_each = True
                    save_stacked = True
                elif fig_idx is not True:
                    min_seq = fig_idx
                    max_seq = fig_idx+1 # Save a specific output sequence
                    save_after_each = True
                    save_stacked = False
                else:
                    min_seq = 0
                    max_seq = self.n_seq 
                    save_after_each = False # Stack all outputs in one plot
                    save_stacked = True

                if save_stacked or compute_mean or compute_mad:
                    fig, ax = plt.subplots()
                    fig.tight_layout()

                for n in range(min_seq, max_seq):
                    if save_after_each:
                        fig_i, ax_i = plt.subplots()
                        fig_i.tight_layout()

                    for x in range(layer_z.shape[2]): # for each dimension
                        x_plot = np.linspace(1, self.max_timesteps+1, self.max_timesteps) # timesteps on X
                        y_plot = layer_z[n, :, x]

                        if save_stacked:
                            if fig_plot_markers:
                                ax.scatter(x_plot, y_plot)
                            ax.plot(x_plot, y_plot)

                        if save_after_each:
                            if fig_plot_markers:
                                ax_i.scatter(x_plot, y_plot)
                            ax_i.plot(x_plot, y_plot)

                    if save_after_each:
                        ax_i.set_xlim(left=0.0)
                        ax_i.set_ylim(bottom=fig_ymin[l-min_layer], top=fig_ymax[l-min_layer])
                        if filename_prefix:
                            figfilename = str(filename_prefix) + '_' + modality + "_z" + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.png"
                        else:
                            figfilename = modality + "_z"  + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.png"  
                        if dir is not None:
                            figfilename = dir + '/' + figfilename
                            if not os.path.exists(dir):
                                os.makedirs(dir)

                        fig_i.savefig(figfilename)
                        plt.close(fig_i)

                if save_stacked or compute_mean or compute_mad:
                    if compute_mean or compute_mad:
                        if compute_mean:
                            y_plot = np.mean(np.mean(layer_z, axis=0), axis=1)
                        elif compute_mad:
                            y_plot = np.sqrt(np.mean(np.linalg.norm(layer_z-np.mean(layer_z, axis=0), axis=2), axis=0)) # mean absolute deviation
                        ax.clear()
                        if fig_plot_markers:
                            ax.scatter(x_plot, y_plot)
                        ax.plot(x_plot, y_plot)
                        if compute_mean:
                            ax.set_ylim(bottom=fig_ymin[l-min_layer], top=fig_ymax[l-min_layer])
                        elif compute_mad:
                            ax.set_ylim(0, 1)
                    else:
                        ax.set_ylim(bottom=fig_ymin[l-min_layer], top=fig_ymax[l-min_layer])
                    ax.set_xlim(left=0.0)
                    if filename_prefix:
                        figfilename = str(filename_prefix) + '_' + modality + "_z" + str(l) + "_e" + str(epoch) + "_out.png"
                    else:
                        figfilename = modality + "_z"  + str(l) + "_e" + str(epoch) + "_out.png"
                    if dir is not None:
                        figfilename = dir + '/' + figfilename
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                    fig.savefig(figfilename)
                    plt.close(fig)

    def write_file_z_src(self, generated, epoch, modality, idx, layer=None, filename_prefix=None, dir=None, fig_idx=True, fig_plot_markers=False, fig_y_lim=None, compute_mad=False):
        # timesteps, layers, batch, z-units (max pad)
        generated = generated[0]
        # Select which layer to save
        if layer is None:
            min_layer = 1
            max_layer = self.n_layers[modality] # Output for each training sequence
        else:
            min_layer = layer
            max_layer = layer+1

        for l in range(min_layer, max_layer):
            # Select which sequence to save
            if idx is None:
                min_seq = 0
                max_seq = self.n_seq # Output for each training sequence
            else:
                min_seq = idx
                max_seq = idx+1
            for n in range(min_seq, max_seq):
                layer_z = generated[:,l,n,:]
                if len(layer_z.shape) > 2:
                    layer_z = np.squeeze(layer_z)
                if filename_prefix:
                    filename_z = str(filename_prefix) + '_' + modality + "_z" + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_src.csv"
                else:
                    filename_z = modality + "_z" + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_src.csv"

                if dir is not None:
                    filename_z = dir + '/' + filename_z
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                
                file_write_z = open(filename_z, 'w')

                ## Initial row
                # Columns for CSV
                num_z = self.z_units[modality][l] if self.vb_posterior_src_extend else self.z_units[modality][l]*2
                for x in range(num_z): # output dims
                    if x == 0:
                        file_write_z.write("zd%d_%d" % (l, x))
                    else:
                        file_write_z.write(",zd%d_%d" % (l, x))
                file_write_z.write("\n")
                # Skip t=0
                for x in range(num_z): # output dims
                    file_write_z.write(",")
                file_write_z.write("\n")

                for step in range(self.max_timesteps):
                    for x in range(num_z):
                        if x == 0:
                            file_write_z.write("%f" % layer_z[step, x])
                        else:
                            file_write_z.write(",%f" % layer_z[step, x])
                    file_write_z.write("\n")
                file_write_z.close()
            
            save_stacked = False
            if fig_idx is not False:
                # Select what and how to plot
                if fig_idx is None:
                    min_seq = 0
                    max_seq = self.n_seq # Output for each training sequence
                    save_after_each = True
                    save_stacked = True
                elif fig_idx is not True:
                    min_seq = fig_idx
                    max_seq = fig_idx+1 # Save a specific output sequence
                    save_after_each = True
                    save_stacked = False
                else:
                    min_seq = 0
                    max_seq = self.n_seq 
                    save_after_each = False # Stack all outputs in one plot
                    save_stacked = True

                if save_stacked or compute_mad:
                    fig, ax = plt.subplots()
                    fig.tight_layout()
                        
                for n in range(min_seq, max_seq):
                    layer_z = generated[:,l,n,:]
                    if len(layer_z.shape) > 2:
                        layer_z = np.squeeze(layer_z)
                    if save_after_each:
                        fig_i, ax_i = plt.subplots()
                        fig_i.tight_layout()

                    for x in range(num_z): # for each dimension
                        x_plot = np.linspace(1, layer_z.shape[0]+1, layer_z.shape[0]) # timesteps on X
                        y_plot = layer_z[:, x]

                        if save_stacked:
                            if fig_plot_markers:
                                ax.scatter(x_plot, y_plot)
                            ax.plot(x_plot, y_plot)

                        if save_after_each:
                            if fig_plot_markers:
                                ax_i.scatter(x_plot, y_plot)
                            ax_i.plot(x_plot, y_plot)

                    if save_after_each:
                        ax_i.set_xlim(left=0.0)
                        if fig_y_lim is not None:
                            if fig_y_lim[0] is not None:
                                ax_i.set_ylim(bottom=fig_y_lim[0])
                            if fig_y_lim[1] is not None:
                                ax_i.set_ylim(top=fig_y_lim[1])
                        if filename_prefix:
                            figfilename = str(filename_prefix) + '_' + modality + "_z" + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_src.png"
                        else:
                            figfilename = modality + "_z" + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_src.png"
                        if dir is not None:
                            figfilename = dir + '/' + figfilename
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                        fig_i.savefig(figfilename)
                        plt.close(fig_i)

                if save_stacked or compute_mad:
                    if compute_mad:
                        layer_z = np.swapaxes(generated[:,l,:,:], 0, 1)
                        y_plot = np.sqrt(np.mean(np.linalg.norm(layer_z-np.mean(layer_z, axis=0), axis=2), axis=0)) # mean absolute deviation
                        ax.clear()
                        if fig_plot_markers:
                            ax.scatter(x_plot, y_plot)
                        ax.plot(x_plot, y_plot)
                        ax.set_ylim((0.0, 2.0))
                    else:
                        if fig_y_lim is not None:
                            if fig_y_lim[0] is not None:
                                ax.set_ylim(bottom=fig_y_lim[0])
                            if fig_y_lim[1] is not None:
                                ax.set_ylim(top=fig_y_lim[1])
                    ax.set_xlim(left=0.0)
                    if filename_prefix:
                        figfilename = str(filename_prefix) + '_' + modality + "_z" + str(l) + "_e" + str(epoch) + "_src.png"
                    else:
                        figfilename = modality + "_z" + str(l) + "_e" + str(epoch) + "_src.png"
                    if dir is not None:
                        figfilename = dir + '/' + figfilename
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                    fig.savefig(figfilename)
                    plt.close()

    def write_file_loss(self, generated, epoch, modality, idx, layer=None, filename_prefix=None, dir=None, fig_idx=True, fig_y_lim=None, fig_plot_markers=False, compute_mean=False, abs_values=True):
        generated = generated[0]
        # Select which layer to save
        if layer is None:
            min_layer = 1
            max_layer = self.n_layers[modality] # Output for each training sequence
        else:
            min_layer = layer
            max_layer = layer+1

        for l in range(min_layer, max_layer):
            layer_l = np.asarray(generated[l])[1]
            if abs_values:
                layer_l = abs(layer_l)

            # Select which sequence to save
            if idx is None:
                min_seq = 0
                max_seq = self.n_seq # Output for each training sequence
            else:
                min_seq = idx
                max_seq = idx+1
            for n in range(min_seq, max_seq):
                if filename_prefix:
                    filename = str(filename_prefix) + '_' + modality + "_l" + str(l) + "_e" + str(epoch) + "_n" + str(n) + ".csv"
                else:
                    filename = modality + "_l" + str(l) + "_e" + str(epoch) + "_n" + str(n) + ".csv"

                if dir is not None:
                    filename = dir + '/' + filename
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                
                file_write = open(filename, 'w')

                ## Initial row
                # Columns for CSV
                file_write.write(filename_prefix)
                file_write.write("\n")

                for step in range(self.max_timesteps): # total timesteps
                    file_write.write("%f\n" % layer_l[n, step])
                file_write.close()

            save_stacked = False
            if fig_idx is not False:
                # Select what and how to plot
                if fig_idx is None:
                    min_seq = 0
                    max_seq = self.n_seq # Output for each training sequence
                    save_after_each = True
                    save_stacked = True
                elif fig_idx is not True:
                    min_seq = fig_idx
                    max_seq = fig_idx+1 # Save a specific output sequence
                    save_after_each = True
                    save_stacked = False
                else:
                    min_seq = 0
                    max_seq = self.n_seq 
                    save_after_each = False # Stack all outputs in one plot
                    save_stacked = True

                if save_stacked or compute_mean:
                    fig, ax = plt.subplots()
                    fig.tight_layout()

                for n in range(min_seq, max_seq):
                    if save_after_each:
                        fig_i, ax_i = plt.subplots()
                        fig_i.tight_layout()

                    x_plot = np.linspace(0, self.max_timesteps, self.max_timesteps) # timesteps on X
                    y_plot = layer_l[n, :]

                    if save_stacked:
                        if fig_plot_markers:
                            ax.scatter(x_plot, y_plot)
                        ax.plot(x_plot, y_plot)

                    if save_after_each:
                        if fig_plot_markers:
                            ax_i.scatter(x_plot, y_plot)
                        ax_i.plot(x_plot, y_plot)
                        ax_i.set_xlim(left=0.0)
                        if fig_y_lim is not None:
                            ax_i.set_ylim(bottom=fig_y_lim[0], top=fig_y_lim[1])
                        if filename_prefix:
                            figfilename = str(filename_prefix) + '_' + modality + "_l" + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.png"
                        else:
                            figfilename = modality + "_l"  + str(l) + "_e" + str(epoch) + "_n" + str(n) + "_out.png"  
                        if dir is not None:
                            figfilename = dir + '/' + figfilename
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                        fig_i.savefig(figfilename)
                        plt.close(fig_i)

                if save_stacked or compute_mean:
                    if compute_mean:
                        y_plot = np.mean(layer_l, axis=0)
                        ax.clear()
                        if fig_plot_markers:
                            ax.scatter(x_plot, y_plot)
                        ax.plot(x_plot, y_plot)
                    ax.set_xlim(left=0.0)
                    if fig_y_lim is not None:
                        ax.set_ylim(bottom=fig_y_lim[0], top=fig_y_lim[1])
                    if filename_prefix:
                        figfilename = str(filename_prefix) + '_' + modality + "_l" + str(l) + "_e" + str(epoch) + "_out.png"
                    else:
                        figfilename = modality + "_l"  + str(l) + "_e" + str(epoch) + "_out.png"
                    if dir is not None:
                        figfilename = dir + '/' + figfilename
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                    fig.savefig(figfilename)
                    plt.close(fig)

    # Tensorflow state
    def save(self, epoch, model_name):
        checkpoint_dir = os.path.join(model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, checkpoint_dir + '/model', global_step=epoch)
        # self.saver.save(self.sess, checkpoint_dir + '/model')
        # print("save: Saved checkpoint (model-%s)" % str(epoch))

    def load(self, model_name, ckpt_name=None):
        checkpoint_dir = os.path.join(model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if ckpt_name is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print("load: Loaded checkpoint {}".format(ckpt_name))
            return (int(re.findall(r'\d+', ckpt_name)[0])) # Return checkpoint ID (epoch)
        else:
            print("load: Failed to load checkpoint for %s" % model_name)
            return -1
