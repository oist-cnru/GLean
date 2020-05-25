import builtins
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import time
import sys
import os
import configparser
import argparse

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
tf_config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.OFF # workaround TF bug
# tf_config.gpu_options.allow_growth = True
# tf_config.graph_options.rewrite_options.auto_mixed_precision = True # enable FP16/FP32 auto optimization for Volta+ GPUs (requires TF version >=1.14)
# tf_config.log_device_placement = True

# training_cl_ratio = 0.9 # no effect on new model
er_start_epoch = 0
# Defaults, overridden later
er_max_epoch = 500
er_learning_rate = 0.1 # increase this for faster plan learning
er_opt_epsilon = 0.01 # increase this to avoid instability from rapid learning
# Report every x epochs
print_epochs = 10
save_epochs = print_epochs

save_planning_model = False
save_fig_idx = 0 # set to none to save all
save_z_layer = None # set to none to save all
save_d_layer = 0 # usually we only save motor layer 0 (output)
save_posterior = True
save_prior = True
# "Lowest loss idx" is the posterior with the lowest reconstruction loss in the current epoch
lowest_loss_idx_refresh = False
lowest_loss_idx = 0
lowest_loss = sys.float_info.max
# "Lowest loss" here is the epoch with the lowest total loss
save_lowest_total_loss_only = True
save_after = 0 # let ER "warmup" before starting to save
save_only_improved = False # skip saving if no improvement in loss
save_full_loss = False
save_candidate_plan_loss = True # log loss values of all candidates
modalities = ["xxxx"]

output_subdir = "/planning_output"
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
        config = configparser.ConfigParser()
        if len(config.read(config_file)) == 0:
            print("Failed to read config file " + config_file)
            exit(1)
        else:
            print("Loaded config " + config_file)
            config_data = dict(config.items("data")) # for the model to load we need the same configuration as in training
            config_network = dict(config.items("network"))
            config_planning = dict(config.items("planning"))
            modalities = [str(m.strip()) for m in config_network["modalities"].split(',')]
else:
    print("Config file required as first argument! Use --help to see other command line options")
    exit(1)

# Handle command line arguments
parser = argparse.ArgumentParser(description="Generate an action plan using a trained RNN. See the readme for usage guidance.")
# Required (but already parsed manually)
parser.add_argument("config_file", help="configuration file defining the model (required as first argument)")
# Optional
parser.add_argument("--checkpoint", help="checkpoint file to restore from", metavar="MODEL-YYYY")
parser.add_argument("--output_dir", help="subdirectory of the training path to save output", metavar="DIR")
parser.add_argument("--print_epochs", help="print status every N epochs", metavar="N", type=int)
parser.add_argument("--save_epochs", help="save a checkpoint every N epochs", metavar="N", type=int)
parser.add_argument("--save_all_epochs", help="convenience flag to set print_epochs and save_epochs to 1", action="store_const", const=1)
parser.add_argument("--save_only_after", help="start saving after N epochs", metavar="N", type=int)
parser.add_argument("--get_lowest_loss", help="select the generated sequence with the lowest loss", action="store_true")
parser.add_argument("--save_all_samples", help="save all generated sequences (causes a lot of I/O overhead)", action="store_false")
parser.add_argument("--save_sample", help="save given sample only", metavar="N", type=int)
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
parser.add_argument("--posterior_src_range", help="only use the posterior for the given timestep range", metavar="A,B")
parser.add_argument("--kld_calc_range", help="only calculate KLD_pq in the given timestep range", dest="overrides_kld_range", metavar="A,B")
parser.add_argument("--disable_ugaussian", help="disable the unit gaussian prior if set in config", action="store_true")
parser.add_argument("--save_full_loss", help="save loss values per timestep per sequence", action="store_true")
# Optional config overrides
parser.add_argument("--training_path", help="path to load the trained model", dest="config_data_training_path", metavar="PATH")
parser.add_argument("--trained_sequences", help="the number of trained sequences", dest="config_data_sequences", metavar="N")
parser.add_argument("--optimizer", help="optimizer to use when training", dest="config_network_optimizer", metavar="FUNC")
parser.add_argument("--learning_rate", help="learning rate", dest="config_planning_learning_rate", metavar="F")
parser.add_argument("--opt_epsilon", help="epsilon value for optimizer", dest="config_planning_opt_epsilon", metavar="F")
parser.add_argument("--gradient_clip", help="gradient clipping", dest="config_network_gradient_clip", metavar="F")
parser.add_argument("--gradient_clip_input", help="input clipping", dest="config_network_gradient_clip_input", metavar="F")
parser.add_argument("--ugaussian_t_range", help="use unit gaussian as prior for given timestep range", dest="config_network_ugaussian_t_range", metavar="A,B")
parser.add_argument("--ugaussian_weight", help="weight applied to unit gaussian loss", dest="config_network_ugaussian_weight", metavar="F")
parser.add_argument("--max_epochs", help="maximum number of epochs to run for", dest="config_planning_max_epochs", metavar="N")
parser.add_argument("--init_frame", help="first frame of the initial frames used for plan generation", dest="config_planning_init_frame", metavar="N")
parser.add_argument("--init_depth", help="number of initial frames for plan generation", dest="config_planning_init_depth", metavar="N")
parser.add_argument("--goal_frame", help="first frame of the goal frames used for plan generation", dest="config_planning_goal_frame", metavar="N")
parser.add_argument("--goal_depth", help="number of goal frames for plan generation", dest="config_planning_goal_depth", metavar="N")
parser.add_argument("--init_frame_duplicate", help="use duplicates of init frame to fill init depth", dest="config_planning_init_frame_duplicate", action="store_const", const="True")
parser.add_argument("--goal_padding", help="use duplicates of goal frame to fill goal depth", dest="config_planning_goal_padding", action="store_const", const="True")
parser.add_argument("--rec_weighting", help="multiplier for weight of missing frames in plan generation", dest="config_planning_rec_weighting", metavar="F")
parser.add_argument("--goal_modalities_mask", help="list defining start and end of data dimensions to use in goal frames", dest="config_planning_goal_modalities_mask", metavar="N")
parser.add_argument("--init_modalities_mask", help="list defining start and end of data dimensions to use in init frames", dest="config_planning_init_modalities_mask", metavar="N")
parser.add_argument("--plot_by_dim", help="a list of dimensions to plot separately", dest="config_planning_plot_by_dim", metavar="A,B,...")
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
        elif key == "output_dir":
            output_subdir = "/" + args["output_dir"]
        elif key == "save_posterior":
            save_posterior = args["save_posterior"]
        elif key == "print_epochs":
            print_epochs = args["print_epochs"]
        elif key == "save_epochs":
            save_epochs = args["save_epochs"]
        elif key == "save_all_epochs":
            print_epochs = 1
            save_epochs = 1
        elif key == "save_only_after":
            save_after = args["save_only_after"]
            save_only_improved = True
        elif key == "save_sample":
            lowest_loss_idx = args[key]
        elif key == "get_lowest_loss" and args[key] == True:
            lowest_loss_idx_refresh = True
        elif key == "save_all_samples":
            save_lowest_total_loss_only = args[key]
            lowest_loss_idx = None
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
        elif key.startswith("config_planning_"):
            config_planning[key[16:]] = args[key]
        elif key.startswith("overrides_"):
            plan_overrides[key[10:]] = args[key]

training_path = config_data["training_path"]

if "fig_xy_dims" in config_planning:
    plot_xy = True
    plot_dims = [int(d.strip()) for d in config_planning["fig_xy_dims"].split(',')]
    plot_x_lim = [float(x.strip()) for x in config_planning["fig_x_lim"].split(',')]
    plot_y_lim = [float(y.strip()) for y in config_planning["fig_y_lim"].split(',')]
if "max_epochs" in config_planning:
    er_max_epoch = int(config_planning["max_epochs"])
if "learning_rate" in config_planning:
    er_learning_rate = float(config_planning["learning_rate"])
elif "learning_rate" in config_network:
    er_learning_rate = float(config_network["learning_rate"])*100
if "opt_epsilon" in config_planning:
    er_opt_epsilon = float(config_planning["opt_epsilon"])
elif "opt_epsilon" in config_network:
    er_opt_epsilon = float(config_network["opt_epsilon"])*100
else:
    er_opt_epsilon = er_learning_rate/10.0 # safe default?

with tf.compat.v1.Session(config = tf_config) as sess:
    rnn = PVRNN(sess, config_data, config_network, learning_rate=er_learning_rate, optimizer_epsilon=er_opt_epsilon, hybrid_posterior_src=override_posterior_src_range, planning=config_planning, reset_posterior_src=True, overrides=plan_overrides)

    print("Load model")
    tf.compat.v1.global_variables_initializer().run()
    epoch = rnn.load(training_path, ckpt_name=training_load_checkpoint)
    if epoch == -1:
        print("Trained model required for planning!")
        exit(255)

    sum_op = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter(training_path + "/er_logs", rnn.sess.graph)

    best_loss = sys.float_info.max
    best_loss_epo = -1

    # Build a list of tensors we want evaluated
    model_epoch = []
    model_epoch_lut = dict()
    model_epoch_idx = 0
    model_epoch.append(rnn.build_model["training_batch"]) # run model
    model_epoch_lut["training_batch"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(sum_op) # summary for tensorboard
    model_epoch_lut["sum_op"] = model_epoch_idx; model_epoch_idx += 1

    # Get loss values
    model_epoch.append(rnn.build_model["total_batch_reconstruction_loss"])
    model_epoch_lut["total_batch_reconstruction_loss"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["total_batch_regularization_loss"])
    model_epoch_lut["total_batch_regularization_loss"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["total_batch_loss"])
    model_epoch_lut["total_batch_loss"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["selected_loss_idx"])
    model_epoch_lut["selected_loss_idx"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["selected_loss"])
    model_epoch_lut["selected_loss"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["selected_loss_rec"])
    model_epoch_lut["selected_loss_rec"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["selected_loss_reg"])
    model_epoch_lut["selected_loss_reg"] = model_epoch_idx; model_epoch_idx += 1

    # Get output
    model_epoch.append(rnn.build_model["initial"])
    model_epoch_lut["initial"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["generated_out"])
    model_epoch_lut["generated_out"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["generated_z_posterior_mean"])
    model_epoch_lut["generated_z_posterior_mean"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["generated_z_posterior_var"])
    model_epoch_lut["generated_z_posterior_var"] = model_epoch_idx; model_epoch_idx += 1
    model_epoch.append(rnn.build_model["generated_z_posterior_src"])
    model_epoch_lut["generated_z_posterior_src"] = model_epoch_idx; model_epoch_idx += 1
    if save_prior:
        model_epoch.append(rnn.build_model["generated_z_prior_mean"])
        model_epoch_lut["generated_z_prior_mean"] = model_epoch_idx; model_epoch_idx += 1
        model_epoch.append(rnn.build_model["generated_z_prior_var"])
        model_epoch_lut["generated_z_prior_var"] = model_epoch_idx; model_epoch_idx += 1
    if save_full_loss:
        model_epoch.append(rnn.build_model["batch_regularization_loss"])
        model_epoch_lut["batch_regularization_loss"] = model_epoch_idx; model_epoch_idx += 1
        model_epoch.append(rnn.build_model["batch_reconstruction_loss"])
        model_epoch_lut["batch_reconstruction_loss"] = model_epoch_idx; model_epoch_idx += 1

    if save_candidate_plan_loss:
        candidate_plan_loss = []
    print("Beginning iterations")
    start_time = time.time()
    for epoch in range(er_start_epoch, er_max_epoch):
        # Gradients debug
        # model_epoch.append(rnn.build_model["gradients"])
        # model_epoch_lut["gradients"] = model_epoch_idx; model_epoch_idx += 1

        # Evaluate for this epoch
        model_epoch_output = sess.run(model_epoch)

        # grads = model_epoch[model_epoch_lut["gradients"]][0].itervalues().next()
        # np.save("grads_" + str(epoch) + ".npy", grads)

        # Parse output
        writer.add_summary(model_epoch_output[model_epoch_lut["sum_op"]], epoch) # summary for TensorBoard

        # Print to terminal
        if (epoch+1) % print_epochs == 0 or (epoch+1) % save_epochs == 0:
            loss_rec = model_epoch_output[model_epoch_lut["total_batch_reconstruction_loss"]]/rnn.build_model["data_length"]
            loss_reg = model_epoch_output[model_epoch_lut["total_batch_regularization_loss"]]/rnn.build_model["data_length"]
            loss_total = model_epoch_output[model_epoch_lut["total_batch_loss"]]/rnn.build_model["data_length"]
            xprint("Epoch %d  time %.0f / %.3f  loss_total %.3f  loss_rec %.3f  loss_reg %.3f" % (epoch, time.time() - start_time, (time.time() - start_time)/(epoch-er_start_epoch+1), loss_total, loss_rec, -loss_reg))
            if np.isnan(loss_total):
                xprint("**Halted**")
                exit(1)

            if lowest_loss_idx_refresh:
                lowest_loss_idx = model_epoch_output[model_epoch_lut["selected_loss_idx"]]
                lowest_loss = model_epoch_output[model_epoch_lut["selected_loss"]]
                if save_candidate_plan_loss:
                    candidate_plan_loss.append((epoch, lowest_loss, model_epoch_output[model_epoch_lut["selected_loss_rec"]], model_epoch_output[model_epoch_lut["selected_loss_reg"]]))
                xprint("Sample with lowest loss: " + str(lowest_loss_idx) + " (" + str(round(lowest_loss, 3)) + ")")
                # if lowest_loss < best_loss:
                    # best_loss = lowest_loss
                if loss_total < best_loss:
                    best_loss = loss_total
                    best_loss_epo = epoch
                xprint("Best plan so far: " + str(best_loss_epo) + " (" + str(round(best_loss, 3)) + ")")
            else:
                if save_candidate_plan_loss:
                    candidate_plan_loss.append((epoch, loss_total, loss_rec, loss_reg))
                if loss_total < best_loss:
                    best_loss = loss_total
                    best_loss_epo = epoch
                xprint("Best overall plan so far: " + str(best_loss_epo) + " (" + str(round(best_loss, 3)) + ")")
        
        if ((epoch+1) % save_epochs == 0 and epoch != 0 and epoch > save_after and (not save_only_improved or best_loss_epo == epoch)) or (not save_only_improved and epoch == er_max_epoch-1):
            for m in modalities:
                if save_planning_model:
                    rnn.save(epoch, training_path)

                motor_initial = [model_epoch_output[model_epoch_lut["initial"]][m]]
                generated_out = [model_epoch_output[model_epoch_lut["generated_out"]][m]]
                generated_out_closed_z_q_mean = [model_epoch_output[model_epoch_lut["generated_z_posterior_mean"]][m]]
                generated_out_closed_z_q_var = [model_epoch_output[model_epoch_lut["generated_z_posterior_var"]][m]]
                generated_out_closed_z_q_src = [model_epoch_output[model_epoch_lut["generated_z_posterior_src"]][m]]
                if save_d_layer != 0:
                    rnn.write_file_csv(generated_out, epoch, m, lowest_loss_idx, save_d_layer, None, training_path + output_subdir, fig_idx=lowest_loss_idx, initial=motor_initial)
                rnn.write_file_decoded(generated_out, epoch, m, lowest_loss_idx, None, training_path + output_subdir, sm_min=float(config_data[m + "_softmax_min"]), sm_max=float(config_data[m + "_softmax_max"]), fig_idx=lowest_loss_idx, fig_xy=plot_xy, fig_plot_dims=plot_dims, fig_x_lim=plot_x_lim, fig_y_lim=plot_y_lim)
                if "plot_by_dim" in config_planning:
                    plot_dims = [int(d.strip()) for d in config_planning["plot_by_dim"].split(',')]
                    for r in plot_dims:
                        rnn.write_file_decoded(generated_out, epoch, m, lowest_loss_idx, "dim" + str(r), training_path + output_subdir, sm_min=float(config_data[m + "_softmax_min"]), sm_max=float(config_data[m + "_softmax_max"]), fig_idx=lowest_loss_idx, fig_plot_dims=[r], skip_csv=True)
                rnn.write_file_z(generated_out_closed_z_q_mean, epoch, m, lowest_loss_idx, save_z_layer, "posterior_mean", training_path + output_subdir, fig_idx=lowest_loss_idx)
                rnn.write_file_z(generated_out_closed_z_q_var, epoch, m, lowest_loss_idx, save_z_layer, "posterior_var", training_path + output_subdir, fig_idx=lowest_loss_idx, fig_ymin=0.0, fig_ymax=2.0)
                rnn.write_file_z_src(generated_out_closed_z_q_src, epoch, m, lowest_loss_idx, save_z_layer, "posterior_src", training_path + output_subdir, fig_idx=lowest_loss_idx, fig_y_lim=[-2.0,2.0])

                if save_prior:
                    generated_out_closed_z_p_mean = [model_epoch_output[model_epoch_lut["generated_z_prior_mean"]][m]]
                    generated_out_closed_z_p_var = [model_epoch_output[model_epoch_lut["generated_z_prior_var"]][m]]
                    rnn.write_file_z(generated_out_closed_z_p_mean, epoch, m, lowest_loss_idx, save_z_layer, "prior_mean", training_path + output_subdir, fig_idx=lowest_loss_idx)
                    rnn.write_file_z(generated_out_closed_z_p_var, epoch, m, lowest_loss_idx, save_z_layer, "prior_var", training_path + output_subdir, fig_idx=lowest_loss_idx, fig_ymin=0.0, fig_ymax=2.0)
                if save_full_loss:
                    full_reg_loss = [model_epoch_output[model_epoch_lut["batch_regularization_loss"]][m]]
                    full_rec_loss = [model_epoch_output[model_epoch_lut["batch_reconstruction_loss"]][m]]
                    rnn.write_file_loss(full_reg_loss, epoch, m, lowest_loss_idx, save_z_layer, "reg_loss", training_path + output_subdir, fig_idx=lowest_loss_idx)
                    rnn.write_file_loss(np.expand_dims(full_rec_loss, axis=0), epoch, m, lowest_loss_idx, 0, "rec_loss", training_path + output_subdir, fig_idx=lowest_loss_idx)
    print("Done!")

if dump_config:
    f = open(training_path + output_subdir + "/settings.txt", 'w')
    ## Dump config
    f.write("**learning args**\n")
    f.write("learning_rate = " + str(er_learning_rate) + "\n")
    f.write("optimizer_epsilon = " + str(er_opt_epsilon) + "\n")
    f.write("**config_data**\n")
    for key in config_data:
        f.write("('" + str(key) + ", '" + str(config_data[key]) + "')\n")
    f.write("**config_network**\n")
    for key in config_network:
        f.write("('" + str(key) + ", '" + str(config_network[key]) + "')\n")
    f.write("**planning**\n")
    for key in config_planning:
        f.write("('" + str(key) + ", '" + str(config_planning[key]) + "')\n")
    f.close()

if save_candidate_plan_loss:
    f = open(training_path + output_subdir + "/candidate_plans.csv", 'w')
    f.write("epoch,loss_total,loss_rec,loss_reg\n")
    for l in candidate_plan_loss:
        f.write(str(l[0]) + "," + str(l[1]) + "," + str(l[2]) + "," + str(-l[3]) + "\n")
    f.close()

    selected = candidate_plan_loss[best_loss_epo//save_epochs]
    f = open(training_path + output_subdir + "/selected_plan.csv", 'w')
    f.write("epoch,loss_total,loss_rec,loss_reg\n")
    f.write(str(selected[0]) + "," + str(selected[1]) + "," + str(selected[2]) + "," + str(-selected[3]))
    f.close()