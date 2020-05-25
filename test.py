import builtins
import numpy as np
import tensorflow as tf
import time
import sys
import os
import configparser
import argparse
import csv

from model import PVRNN
import ops

# Override default print
def print(*args, **kwargs):
    builtins.print(sys.argv[0][:-3], end = ": ")
    return builtins.print(*args, **kwargs)

def xprint(*args, **kwargs):
    return builtins.print(*args, **kwargs)

NUM_CPU = 4
NUM_GPU = 0
# MKL params
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=NUM_CPU, inter_op_parallelism_threads=NUM_CPU, device_count={"GPU": NUM_GPU, "CPU": NUM_CPU}) # use CPU only
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# tf_config.graph_options.rewrite_options.auto_mixed_precision = True # enable FP16/FP32 auto optimization for Volta+ GPUs (requires TF version >=1.14)
# tf_config.log_device_placement = True

save_seq_idx = None # Set to none to save all
save_fig_idx = None # Set to none to save all, true to save only stacked
save_raw_z = False
save_z_src = True
save_raw_d = True
save_myu_sigma = True
save_z_layer = None # Set to none to save all
save_d_layer = None # Raw output (d) of neurons
save_posterior = True
save_prior = True
save_full_loss = False
modalities = ["xxxx"]

output_subdir = "/testing_output"
training_load_checkpoint = None
plot_xy = False
plot_dims = None
plot_x_lim = None
plot_y_lim = None
plot_by_dim = True
dump_config = True # save the settings used in the training directory for later reference
override_d_output = None
override_posterior_src_range = None
override_posterior_src_idx = None
test_overrides = dict()
prior_generation = False
reset_posterior = False

# Load data and network info from config file first
if len(sys.argv) > 1:
    if sys.argv[1] != "-h" and sys.argv[1] != "--help":
        config_file = sys.argv[1]
        config = configparser.ConfigParser()
        if len(config.read(config_file)) == 0:
            print("Failed to read config file " + config_file)
            exit(1)
        else:
            print("Loaded config " + config_file)
            config_data = dict(config.items("data")) # for the model to load we need the same configuration as in training
            config_network = dict(config.items("network"))
            if config.has_section("testing"):
                config_testing = dict(config.items("testing"))
            else:
                config_testing = dict()
            modalities = [str(m.strip()) for m in config_network["modalities"].split(',')]
else:
    print("Config file required as first argument! Use --help to see other command line options")
    exit(1)

# Handle command line arguments
parser = argparse.ArgumentParser(description="Generate output from a trained a RNN. See the readme for usage guidance.")
# Required (but already parsed manually)
parser.add_argument("config_file", help="configuration file defining the model (required as first argument)")
# Optional
parser.add_argument("--checkpoint", help="checkpoint file to restore from", metavar="MODEL-YYYY")
parser.add_argument("--output_dir", help="subdirectory of the training path to save output", metavar="DIR")
parser.add_argument("--prior_only", help="use prior Z for generation and reset posterior src", action="store_true")
parser.add_argument("--prior_generation", help="use prior Z for generation", action="store_true")
parser.add_argument("--plot_z_mean", help="plot mean (or mean absolute difference) over all z and sequences", action="store_true")
parser.add_argument("--plot_d_diversity", help="plot RMS and AED of each D compared to fixed D (requires override_d)", action="store_true")
parser.add_argument("--posterior_src_range", help="start and end point of posterior generation", metavar="A,B")
parser.add_argument("--posterior_src_idx", help="use a specific trained A index", type=int, metavar="N")
parser.add_argument("--hybrid_prior_override", help="always posterior instead of prior when generating with given A", action="store_true", dest="overrides_hybrid_posterior_override")
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
parser.add_argument("--disable_ugaussian", help="disable the unit gaussian prior if set in config", action="store_true")
parser.add_argument("--save_full_loss", help="save loss values per timestep per sequence", action="store_true")
parser.add_argument("--load_training_data", help="load training data in order to calculate reconstruction loss", action="store_true", dest="overrides_load_training_data")
# Optional config overrides
parser.add_argument("--training_path", help="path to save the model", dest="config_data_training_path", metavar="PATH")
parser.add_argument("--trained_sequences", help="the number of trained sequences", dest="config_data_sequences", metavar="N")
parser.add_argument("--gradient_clip", help="gradient clipping", dest="config_network_gradient_clip", metavar="F")
parser.add_argument("--gradient_clip_input", help="input clipping", dest="config_network_gradient_clip_input", metavar="F")
parser.add_argument("--ugaussian_t_range", help="use unit gaussian as prior for given timestep range", dest="config_network_ugaussian_t_range", metavar="A,B")
parser.add_argument("--override_d_range", help="use reloaded D values in given timestep range", dest="config_testing_override_d_range", metavar="A,B")
parser.add_argument("--plot_by_dim", help="a list of dimensions to plot separately", dest="config_testing_plot_by_dim", metavar="A,B,...")
# Optional per modality config overrides
for m in modalities:
    parser.add_argument("--" + m + "_datapath", help="path to data file for modality " + m, dest="config_data_"+m+"_path", metavar="PATH")
    parser.add_argument("--" + m + "_activation_func", help="activation function for modality " + m, dest="config_network_"+m+"_activation_func", metavar="FUNC")
    parser.add_argument("--" + m + "_meta_prior", help="list of meta priors per layer for modality " + m, dest="config_network_"+m+"_meta_prior", metavar="W")
    parser.add_argument("--" + m + "_layers_neurons", help="list of number of neurons per layer for modality " + m, dest="config_network_"+m+"_layers_neurons", metavar="D")
    parser.add_argument("--" + m + "_layers_z_units", help="list of number of z-units per layer for modality " + m, dest="config_network_"+m+"_layers_z_units", metavar="Z")
    parser.add_argument("--" + m + "_layers_param", help="list of per layer parameters for modality " + m, dest="config_network_"+m+"_layers_param", metavar="T")
    parser.add_argument("--" + m + "_override_d", help="reload saved D values per layer for modality " + m, dest="config_testing_"+m+"_override_d", metavar="D_FILES")

# Parse command line arguments
args = vars(parser.parse_args())
for key in args:
    if args[key] is not None:
        # Local vars
        if key == "checkpoint":
            training_load_checkpoint = args["checkpoint"]
        elif key == "output_dir":
            output_subdir = "/" + args["output_dir"]
        elif key == "prior_only" and args[key] == True:
            prior_generation = True
            reset_posterior = True
            save_posterior = False
        elif key == "prior_generation" and args[key] == True:
            prior_generation = True
        elif key == "posterior_src_range":
            override_posterior_src_range = [int(m.strip()) for m in args["posterior_src_range"].split(',')]
        elif key == "disable_ugaussian" and args[key] == True:
            config_network.pop("ugaussian_t_range", None)
            config_network.pop("ugaussian_weight", None)
        elif key == "posterior_src_idx":
            override_posterior_src_idx = args["posterior_src_idx"]
        elif key == "save_full_loss" and args[key] == True:
            save_full_loss = args[key]
            config_network["return_full_loss"] = save_full_loss # don't set this in the config file
        # Data overrides
        elif key.startswith("config_data_"):
            config_data[key[12:]] = args[key]
        elif key.startswith("config_network_"):
            config_network[key[15:]] = args[key]
        elif key.startswith("config_testing_"):
            config_testing[key[15:]] = args[key]
        elif key.startswith("overrides_"):
            test_overrides[key[10:]] = args[key]

training_path = config_data["training_path"]

for m in modalities:
    if not test_overrides["load_training_data"]:
        config_data[m + "_path"] = None # no input data for testing
    if m + "_override_d" in config_testing:
        override_d_values = [f.strip() for f in config_testing[m + "_override_d"].split(',')]
        if override_d_output is None:
            if "override_d_range" in config_testing:
                override_d_output = [[int(d.strip()) for d in config_testing["override_d_range"].split(',')]]
            else:
                override_d_output = [[0, int(config_data["max_timesteps"])]]
            override_d_output.append({m: len(override_d_values)*[None] for m in modalities})
        for i in range(len(override_d_values)):
            with open(override_d_values[i], 'rb') as csvfile:
                d_values = csv.reader(csvfile, delimiter=',')
                d_values_list = []
                for row in d_values:
                    d_values_list.append(row)
            del d_values_list[0] # Remove header
            if i == 0:
                del d_values_list[0] # Remove empty row
            override_d_output[1][m][i] = np.array(d_values_list, dtype=np.float32)
test_overrides["override_d_output"] = override_d_output

if "fig_xy_dims" in config_testing:
    plot_xy = True
    plot_dims = [int(d.strip()) for d in config_testing["fig_xy_dims"].split(',')]
    plot_x_lim = [float(x.strip()) for x in config_testing["fig_x_lim"].split(',')]
    plot_y_lim = [float(y.strip()) for y in config_testing["fig_y_lim"].split(',')]

with tf.compat.v1.Session(config = tf_config) as sess:
    rnn = PVRNN(sess, config_data, config_network, training=False, prior_generation=prior_generation, reset_posterior_src=reset_posterior, hybrid_posterior_src=override_posterior_src_range, hybrid_posterior_src_idx=override_posterior_src_idx, overrides=test_overrides)
    print("Loading model")
    tf.compat.v1.global_variables_initializer().run()
    epoch = rnn.load(training_path, ckpt_name=training_load_checkpoint)
    if epoch == -1:
        print("Trained model required for testing!")
        exit(255)
    # Build a list of tensors we want evaluated
    model_eval = []
    model_eval_lut = dict()
    model_eval_idx = 0
    model_eval.append(rnn.build_model["initial"])
    model_eval_lut["initial"] = model_eval_idx; model_eval_idx += 1
    model_eval.append(rnn.build_model["generated_out"])
    model_eval_lut["generated_out"] = model_eval_idx; model_eval_idx += 1

    if save_posterior:
        if save_raw_z:
            model_eval.append(rnn.build_model["generated_z_posterior"])
            model_eval_lut["generated_z_posterior"] = model_eval_idx; model_eval_idx += 1
        if save_myu_sigma:
            model_eval.append(rnn.build_model["generated_z_posterior_mean"])
            model_eval_lut["generated_z_posterior_mean"] = model_eval_idx; model_eval_idx += 1
            model_eval.append(rnn.build_model["generated_z_posterior_var"])
            model_eval_lut["generated_z_posterior_var"] = model_eval_idx; model_eval_idx += 1
        if save_z_src:
            model_eval.append(rnn.build_model["generated_z_posterior_src"])
            model_eval_lut["generated_z_posterior_src"] = model_eval_idx; model_eval_idx += 1
    if save_prior:
        if save_raw_z:
            model_eval.append(rnn.build_model["generated_z_prior"])
            model_eval_lut["generated_z_prior"] = model_eval_idx; model_eval_idx += 1
        if save_myu_sigma:
            model_eval.append(rnn.build_model["generated_z_prior_mean"])
            model_eval_lut["generated_z_prior_mean"] = model_eval_idx; model_eval_idx += 1
            model_eval.append(rnn.build_model["generated_z_prior_var"])
            model_eval_lut["generated_z_prior_var"] = model_eval_idx; model_eval_idx += 1

    model_eval.append(rnn.build_model["total_batch_regularization_loss"])
    model_eval_lut["total_batch_regularization_loss"] = model_eval_idx; model_eval_idx += 1
    model_eval.append(rnn.build_model["total_batch_reconstruction_loss"])
    model_eval_lut["total_batch_reconstruction_loss"] = model_eval_idx; model_eval_idx += 1
    model_eval.append(rnn.build_model["total_batch_loss"])
    model_eval_lut["total_batch_loss"] = model_eval_idx; model_eval_idx += 1
    if save_full_loss:
        model_eval.append(rnn.build_model["batch_regularization_loss"])
        model_eval_lut["batch_regularization_loss"] = model_eval_idx; model_eval_idx += 1
        model_eval.append(rnn.build_model["batch_reconstruction_loss"])
        model_eval_lut["batch_reconstruction_loss"] = model_eval_idx; model_eval_idx += 1

    print("Running model in closed loop")
    start_time = time.time()

    model_eval_output = sess.run(model_eval) # run once

    # Parse output
    for m in modalities:
        generated_initial = [model_eval_output[model_eval_lut["initial"]][m]]
        generated_out = [model_eval_output[model_eval_lut["generated_out"]][m]]
        rnn.write_file_decoded(generated_out, epoch, m, save_seq_idx, None, training_path + output_subdir, sm_min=float(config_data[m + "_softmax_min"]), sm_max=float(config_data[m + "_softmax_max"]), fig_idx=save_fig_idx, fig_xy=plot_xy, fig_plot_dims=plot_dims, fig_x_lim=plot_x_lim, fig_y_lim=plot_y_lim, override_d=(override_d_output[1] if args["plot_d_diversity"] and override_d_output is not None else None))
        if "plot_by_dim" in config_testing:
            plot_dims = [int(d.strip()) for d in config_testing["plot_by_dim"].split(',')]
            for r in plot_dims:
                rnn.write_file_decoded(generated_out, epoch, m, save_seq_idx, "dim" + str(r), training_path + output_subdir, sm_min=float(config_data[m + "_softmax_min"]), sm_max=float(config_data[m + "_softmax_max"]), fig_idx=save_fig_idx, fig_plot_dims=[r], override_d=(override_d_output[1] if args["plot_d_diversity"] and override_d_output is not None else None), skip_csv=True)
        if save_raw_d:
            rnn.write_file_csv(generated_out, epoch, m, save_seq_idx, save_d_layer, None, training_path + output_subdir, fig_idx=save_fig_idx, initial=generated_initial, override_d=(override_d_output[1] if args["plot_d_diversity"] and override_d_output is not None else None))

        if save_posterior:
            if save_raw_z:
                generated_out_closed_z_q_mean = [model_eval_output[model_eval_lut["generated_z_posterior"]][m]]
                rnn.write_file_z(generated_out_closed_z_q_mean, epoch, m, save_seq_idx, save_z_layer, "posterior", training_path + output_subdir, fig_idx=save_fig_idx, compute_entropy=True)
            if save_myu_sigma:
                generated_out_closed_z_q_mean = [model_eval_output[model_eval_lut["generated_z_posterior_mean"]][m]]
                generated_out_closed_z_q_var = [model_eval_output[model_eval_lut["generated_z_posterior_var"]][m]]
                rnn.write_file_z(generated_out_closed_z_q_mean, epoch, m, save_seq_idx, save_z_layer, "posterior_mean", training_path + output_subdir, fig_idx=save_fig_idx, compute_mad=args["plot_z_mean"])
                rnn.write_file_z(generated_out_closed_z_q_var, epoch, m, save_seq_idx, save_z_layer, "posterior_var", training_path + output_subdir, fig_idx=save_fig_idx, fig_ymin=0.0, fig_ymax=2.0, compute_mean=args["plot_z_mean"])
            if save_z_src:
                generated_out_closed_z_q_src = [model_eval_output[model_eval_lut["generated_z_posterior_src"]][m]]
                rnn.write_file_z_src(generated_out_closed_z_q_src, epoch, m, save_seq_idx, save_z_layer, "posterior_src", training_path + output_subdir, fig_idx=save_fig_idx, fig_y_lim=[-2.0,2.0], compute_mad=args["plot_z_mean"])
        if save_prior:
            if save_raw_z:
                generated_out_closed_z_p_mean = [model_eval_output[model_eval_lut["generated_z_prior"]][m]]
                rnn.write_file_z(generated_out_closed_z_p_mean, epoch, m, save_seq_idx, save_z_layer, "prior", training_path + output_subdir, fig_idx=save_fig_idx, compute_entropy=True)
            if save_myu_sigma:
                generated_out_closed_z_p_mean = [model_eval_output[model_eval_lut["generated_z_prior_mean"]][m]]
                generated_out_closed_z_p_var = [model_eval_output[model_eval_lut["generated_z_prior_var"]][m]]
                rnn.write_file_z(generated_out_closed_z_p_mean, epoch, m, save_seq_idx, save_z_layer, "prior_mean", training_path + output_subdir, fig_idx=save_fig_idx, compute_mad=args["plot_z_mean"])
                rnn.write_file_z(generated_out_closed_z_p_var, epoch, m, save_seq_idx, save_z_layer, "prior_var", training_path + output_subdir, fig_idx=save_fig_idx, fig_ymin=0.0, fig_ymax=2.0, compute_mean=args["plot_z_mean"])
        if save_full_loss:
            full_reg_loss = [model_eval_output[model_eval_lut["batch_regularization_loss"]][m]]
            full_rec_loss = [model_eval_output[model_eval_lut["batch_reconstruction_loss"]][m]]
            rnn.write_file_loss(full_reg_loss, epoch, m, save_seq_idx, save_z_layer, "reg_loss", training_path + output_subdir, fig_idx=save_fig_idx, fig_y_lim=[0,None], compute_mean=args["plot_z_mean"])
            rnn.write_file_loss(np.expand_dims(full_rec_loss, axis=0), epoch, m, save_seq_idx, 0, "rec_loss", training_path + output_subdir, fig_idx=save_fig_idx, fig_y_lim=[0,None], compute_mean=args["plot_z_mean"])

    loss_total = model_eval_output[model_eval_lut["total_batch_loss"]]/rnn.build_model["data_length"]
    loss_reg = model_eval_output[model_eval_lut["total_batch_regularization_loss"]]/rnn.build_model["data_length"]
    loss_rec = model_eval_output[model_eval_lut["total_batch_reconstruction_loss"]]/rnn.build_model["data_length"]
    print("loss_total %.3f  loss_rec %.3f  loss_reg %.3f" % (loss_total, loss_rec, -loss_reg))
    print("Done! (time elapsed %.2fs)" % ((time.time()-start_time)))

if dump_config:
    f = open(training_path + output_subdir + "/settings.txt", 'w')
    ## Dump config
    f.write("**generation args**\n")
    f.write("prior_generation = " + str(prior_generation) + "\n")
    f.write("reset_posterior_src = " + str(reset_posterior) + "\n")
    f.write("hybrid_posterior_src = " + str(override_posterior_src_range) + "\n")
    f.write("hybrid_posterior_src_idx = " + str(override_posterior_src_idx) + "\n")
    f.write("**config_data**\n")
    for key in config_data:
        f.write("('" + str(key) + ", '" + str(config_data[key]) + "')\n")
    f.write("**config_network**\n")
    for key in config_network:
        f.write("('" + str(key) + ", '" + str(config_network[key]) + "')\n")
    f.write("**overrides**\n")
    for key in test_overrides:
        f.write("('" + str(key) + ", '" + str(test_overrides[key]) + "')\n")
    f.close()