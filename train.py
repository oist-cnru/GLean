from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import __builtin__
import numpy as np
import tensorflow as tf
import time
import sys
import os
import ConfigParser
import argparse

from model import PVRNN
import ops

# Override default print
def print(*args, **kwargs):
    __builtin__.print(sys.argv[0][:-3], end = ": ")
    return __builtin__.print(*args, **kwargs)

def xprint(*args, **kwargs):
    return __builtin__.print(*args, **kwargs)

NUM_CPU = 4
NUM_GPU = 0
# MKL params
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
tf_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=NUM_CPU, inter_op_parallelism_threads=NUM_CPU, device_count={"GPU": NUM_GPU, "CPU": NUM_CPU}) # use CPU only
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# tf_config.graph_options.rewrite_options.auto_mixed_precision = True # enable FP16/FP32 auto optimization for Volta+ GPUs (requires TF version >=1.14)
# tf_config.log_device_placement = True

training_load = True # load last checkpoint if available
training_start_epoch = 0
# Defaults, overridden later
training_max_epoch = 50000
training_learning_rate = 0.001
training_opt_epsilon = 0.0001
# Report every x epochs
print_epochs = 100
save_epochs = 1000

save_seq_idx = 0 # set to none to save all
save_z_layer = None # set to none to save all
save_d_layer = None # usually we only save motor layer 0 (output)
save_posterior = True
save_reg = True
save_z_src = True
save_full_loss = False
modalities = ["xxxx"]

output_split = [0]
output_scaling = [1.0]
training_load_checkpoint = None
plot_xy = False
plot_dims = None
plot_x_lim = None
plot_y_lim = None
dump_config = True # save the settings used in the training directory for later reference
override_posterior_src_range = None

# Load data and network info from config file first
if len(sys.argv) > 1:
    if sys.argv[1] != "-h" and sys.argv[1] != "--help":
        config_file = sys.argv[1]
        config = ConfigParser.ConfigParser()
        if len(config.read(config_file)) == 0:
            print("Failed to read config file " + config_file)
            exit(1)
        else:
            print("Loaded config " + config_file)
            config_data = dict(config.items("data")) # for the model to load we need the same configuration as in training
            config_network = dict(config.items("network"))
            if config.has_section("training"):
                config_training = dict(config.items("training"))
            else:
                config_training = dict()
            modalities = [str(m.strip()) for m in config_network["modalities"].split(',')]
else:
    print("Config file required as first argument! Use --help to see other command line options")
    exit(1)

# Handle command line arguments
parser = argparse.ArgumentParser(description="Train a RNN. See the readme for usage guidance.")
# Required (but already parsed manually)
parser.add_argument("config_file", help="configuration file defining the model (required as first argument)")
# Optional
parser.add_argument("--checkpoint", help="checkpoint file to restore from", metavar="MODEL-YYYY")
parser.add_argument("--skip_z_output", help="skip output of posterior during training", action="store_false", dest="save_posterior")
parser.add_argument("--print_epochs", help="print status every N epochs", metavar="N", type=int)
parser.add_argument("--save_epochs", help="save a checkpoint every N epochs", metavar="N", type=int)
parser.add_argument("--save_all_epochs", help="convenience flag to set print_epochs and save_epochs to 1", action="store_const", const=1)
parser.add_argument("--save_full_loss", help="save loss values per timestep per sequence", action="store_true") # only provide this as a command line argument
parser.add_argument("--posterior_src_range", help="only use the posterior for the given timestep range", metavar="A,B")
parser.add_argument("--prior_override_l", help="switch to override prior Z calculations - provide a list of layers or keyword all", dest="overrides_prior_override_l", metavar="all|A,B,...")
parser.add_argument("--prior_override_t_range", help="override prior Z calculations for given timestep range", dest="overrides_prior_override_t_range", metavar="A,B")
parser.add_argument("--prior_override_sigma", help="override prior Z sigma with fixed value", dest="overrides_prior_override_sigma", metavar="F")
parser.add_argument("--prior_override_myu", help="override prior Z myu with fixed value", dest="overrides_prior_override_myu", metavar="F")
parser.add_argument("--prior_override_epsilon", help="override prior Z epsilon with fixed value", dest="overrides_prior_override_epsilon", metavar="F")
parser.add_argument("--posterior_override_l", help="switch to override posterior Z calculations - provide a list of layers or keyword all", dest="overrides_posterior_override_l", metavar="all|A,B,...")
parser.add_argument("--posterior_override_t_range", help="override posterior Z calculations for given timestep range", dest="overrides_posterior_override_t_range", metavar="A,B")
parser.add_argument("--posterior_override_sigma", help="override posterior Z sigma with fixed value", dest="overrides_posterior_override_sigma", metavar="F")
parser.add_argument("--posterior_override_myu", help="override posterior Z myu with fixed value", dest="overrides_posterior_override_myu", metavar="F")
parser.add_argument("--posterior_override_epsilon", help="override posterior Z epsilon with fixed value", dest="overrides_posterior_override_epsilon", metavar="F")
parser.add_argument("--kld_calc_range", help="only calculate KLD_pq in the given timestep range", dest="overrides_kld_range", metavar="A,B")
parser.add_argument("--disable_ugaussian", help="disable the unit gaussian prior if set in config", action="store_true")
# Optional config overrides
parser.add_argument("--training_path", help="path to save the model", dest="config_data_training_path", metavar="PATH")
parser.add_argument("--trained_sequences", help="the number of trained sequences", dest="config_data_sequences", metavar="N")
parser.add_argument("--optimizer", help="optimizer to use when training", dest="config_network_optimizer", metavar="FUNC")
parser.add_argument("--learning_rate", help="learning rate", dest="config_network_learning_rate", metavar="F")
parser.add_argument("--opt_epsilon", help="epsilon value for optimizer", dest="config_network_opt_epsilon", metavar="F")
parser.add_argument("--gradient_clip", help="gradient clipping", dest="config_network_gradient_clip", metavar="F")
parser.add_argument("--gradient_clip_input", help="input clipping", dest="config_network_gradient_clip_input", metavar="F")
parser.add_argument("--ugaussian_t_range", help="use unit gaussian as prior for given timestep range", dest="config_network_ugaussian_t_range", metavar="A,B")
parser.add_argument("--ugaussian_weight", help="weight applied to unit gaussian loss", dest="config_network_ugaussian_weight", metavar="F")
parser.add_argument("--max_epochs", help="maximum number of epochs to train for", dest="config_training_max_epochs", metavar="N")
# Optional per modality config overrides
for m in modalities:
    parser.add_argument("--" + m + "_datapath", help="path to data file for modality " + m, dest="config_data_"+m+"_path", metavar="PATH")
    parser.add_argument("--" + m + "_activation_func", help="activation function for modality " + m, dest="config_network_"+m+"_activation_func", metavar="FUNC")
    parser.add_argument("--" + m + "_meta_prior", help="list of meta priors per layer for modality " + m, dest="config_network_"+m+"_meta_prior", metavar="W")
    parser.add_argument("--" + m + "_layers_neurons", help="list of number of neurons per layer for modality " + m, dest="config_network_"+m+"_layers_neurons", metavar="D")
    parser.add_argument("--" + m + "_layers_z_units", help="list of number of z-units per layer for modality " + m, dest="config_network_"+m+"_layers_z_units", metavar="Z")
    parser.add_argument("--" + m + "_layers_param", help="list of per layer parameters for modality " + m, dest="config_network_"+m+"_layers_param", metavar="T")
    parser.add_argument("--" + m + "_seq_prior", help="(experimental) list setting which layers use sequential prior Z calculation for modality " + m, dest="config_network_"+m+"_seq_prior", metavar="B")

# Parse command line arguments
args = vars(parser.parse_args())
plan_overrides = dict()
for key in args:
    if args[key] is not None:
        # Local vars
        if key == "checkpoint":
            training_load_checkpoint = args["checkpoint"]
        elif key == "save_posterior":
            save_posterior = args["save_posterior"]
        elif key == "print_epochs":
            print_epochs = args["print_epochs"]
        elif key == "save_epochs":
            save_epochs = args["save_epochs"]
        elif key == "save_all_epochs":
            print_epochs = 1
            save_epochs = 1
        elif key == "save_full_loss":
            save_full_loss = args[key]
            config_network["return_full_loss"] = save_full_loss # don't set this in the config file
        elif key == "posterior_src_range":
            override_posterior_src_range = [int(m.strip()) for m in args["posterior_src_range"].split(',')]
        elif key == "disable_ugaussian" and args[key] == True:
            config_network.pop("ugaussian_t_range", None)
            config_network.pop("ugaussian_weight", None)
        # Data overrides
        elif key.startswith("config_data_"):
            config_data[key[12:]] = args[key]
        elif key.startswith("config_network_"):
            config_network[key[15:]] = args[key]
        elif key.startswith("config_training_"):
            config_training[key[16:]] = args[key]
        elif key.startswith("overrides_"):
            plan_overrides[key[10:]] = args[key]

training_path = config_data["training_path"]

if "fig_xy_dims" in config_training:
    plot_xy = True
    plot_dims = [int(d.strip()) for d in config_training["fig_xy_dims"].split(',')]
    plot_x_lim = [float(x.strip()) for x in config_training["fig_x_lim"].split(',')]
    plot_y_lim = [float(y.strip()) for y in config_training["fig_y_lim"].split(',')]
if "split" in config_training:
    for s in config_training["split"].split(','):
        output_split.append(int(s.strip()))
    output_scaling = [float(s.strip()) for s in config_training["decode_scaling"].split(',')]
if "max_epochs" in config_training:
    training_max_epoch = int(config_training["max_epochs"])
output_split.append(-1)
if "learning_rate" in config_network:
    training_learning_rate = float(config_network["learning_rate"])
if "opt_epsilon" in config_network:
    training_opt_epsilon = float(config_network["opt_epsilon"])
else:
    training_opt_epsilon = training_learning_rate/10.0 # safe default?

with tf.Session(config = tf_config) as sess:
    rnn = PVRNN(sess, config_data, config_network, learning_rate=training_learning_rate, optimizer_epsilon=training_opt_epsilon, hybrid_posterior_src=override_posterior_src_range, overrides=plan_overrides, data_masking=False)

    print("Start training")
    tf.global_variables_initializer().run()
    if training_load:
        training_start_epoch = rnn.load(training_path, ckpt_name=training_load_checkpoint)
        if training_start_epoch < 0:
            training_start_epoch = 0

    sum_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(training_path + "/logs", rnn.sess.graph)

    ## Build a list of tensors we want evaluated
    model_epoch = []
    model_epoch_lut = dict()
    model_epoch_idx = 0
    model_epoch.append(rnn.build_model["training_batch"]) # run model
    model_epoch_lut["training_batch"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(sum_op) # summary for tensorboard
    model_epoch_lut["sum_op"] = model_epoch_idx; model_epoch_idx += 1

    # Get loss values
    model_epoch.append(rnn.build_model["total_batch_regularization_loss"])
    model_epoch_lut["total_batch_regularization_loss"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["total_batch_reconstruction_loss"])
    model_epoch_lut["total_batch_reconstruction_loss"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["total_batch_loss"])
    model_epoch_lut["total_batch_loss"] = model_epoch_idx; model_epoch_idx += 1

    # Get output
    model_epoch.append(rnn.build_model["generated_out"])
    model_epoch_lut["generated_out"] = model_epoch_idx; model_epoch_idx += 1
    if save_posterior:
        if save_z_src:
            model_epoch.append(rnn.build_model["generated_z_posterior_src"])
            model_epoch_lut["generated_z_posterior_src"] = model_epoch_idx; model_epoch_idx += 1
    if save_full_loss:
        model_epoch.append(rnn.build_model["batch_regularization_loss"])
        model_epoch_lut["batch_regularization_loss"] = model_epoch_idx; model_epoch_idx += 1
        model_epoch.append(rnn.build_model["batch_reconstruction_loss"])
        model_epoch_lut["batch_reconstruction_loss"] = model_epoch_idx; model_epoch_idx += 1

    print("Beginning iterations")
    start_time = time.time()

    for epoch in xrange(training_start_epoch, training_max_epoch):
        # Evaluate for this epoch
        model_epoch_output = sess.run(model_epoch)

        ## Parse output
        writer.add_summary(model_epoch_output[model_epoch_lut["sum_op"]], epoch) # summary for TensorBoard

        # Print loss
        if (epoch+1) % print_epochs == 0 and epoch != 0:
            loss_total = model_epoch_output[model_epoch_lut["total_batch_loss"]]/rnn.build_model["data_length"]
            loss_rec = model_epoch_output[model_epoch_lut["total_batch_reconstruction_loss"]]/rnn.build_model["data_length"]
            if save_reg:
                loss_reg = model_epoch_output[model_epoch_lut["total_batch_regularization_loss"]]/rnn.build_model["data_length"]
            else:
                loss_reg = 0.0
            xprint("Epoch %d  time %.0f / %.3f  loss_total %.3f  loss_rec %.3f  loss_reg %.3f" % (epoch, time.time() - start_time, (time.time() - start_time)/(epoch-training_start_epoch+1), loss_total, loss_rec, -loss_reg))
            if np.isnan(loss_reg) or np.isnan(loss_rec) or loss_rec < 0.0 or -loss_reg < 0.0:
                xprint("**Halted**")
                exit(1)

        # Save network state and some generated results
        if (epoch+1) % save_epochs == 0 and epoch != 0:
            rnn.save(epoch, training_path)
            for m in modalities:
                mscale = m+"_scale_path"
                if mscale in config_data:
                    scale_path = config_data[mscale]
                else:
                    scale_path = None
                generated_out = [model_epoch_output[model_epoch_lut["generated_out"]][m]]
                rnn.write_file_decoded(generated_out, epoch, m, save_seq_idx, None, training_path + "/training_output", sm_min=float(config_data[m + "_softmax_min"]), sm_max=float(config_data[m + "_softmax_max"]), fig_idx=True, fig_xy=plot_xy, fig_plot_dims=plot_dims, fig_x_lim=plot_x_lim, fig_y_lim=plot_y_lim, scale=scale_path) #fig_xy=True, fig_plot_dims=[0,1], fig_x_lim=[0.0, 80.0], fig_y_lim=[40.0, 120.0], scale=scale_path)
                if save_posterior:
                    if save_z_src:
                        generated_out_closed_z_q_src = [model_epoch_output[model_epoch_lut["generated_z_posterior_src"]][m]]
                        rnn.write_file_z_src(generated_out_closed_z_q_src, epoch, m, save_seq_idx, save_z_layer, "posterior_src", training_path + "/training_output", fig_idx=True)
                if save_full_loss:
                    full_reg_loss = [model_epoch_output[model_epoch_lut["batch_regularization_loss"]][m]]
                    full_rec_loss = [model_epoch_output[model_epoch_lut["batch_reconstruction_loss"]][m]]
                    rnn.write_file_loss(full_reg_loss, epoch, m, save_seq_idx, save_z_layer, "reg_loss", training_path + "/training_output", fig_idx=save_seq_idx)
                    rnn.write_file_loss(np.expand_dims(full_rec_loss, axis=0), epoch, m, save_seq_idx, 0, "rec_loss", training_path + "/training_output", fig_idx=save_seq_idx)

    print("Done! (time elapsed %.2fs)" % ((time.time()-start_time)))

if dump_config:
    f = open(training_path + "/training_output/settings.txt", 'w')
    ## Dump config
    f.write("**learning args**\n")
    f.write("learning_rate = " + str(training_learning_rate) + "\n")
    f.write("optimizer_epsilon = " + str(training_opt_epsilon) + "\n")
    f.write("**config_data**\n")
    for key in config_data:
        f.write("('" + str(key) + ", '" + str(config_data[key]) + "')\n")
    f.write("**config_network**\n")
    for key in config_network:
        f.write("('" + str(key) + ", '" + str(config_network[key]) + "')\n")
    f.close()
