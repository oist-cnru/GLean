from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from BasicRNNCell import _linear

## Setting initial states
def reset_neuron_states(batch, neurons):
    # c = tf.random_normal([batch, neurons])
    c = tf.zeros([batch, neurons])
    # h = tf.random_normal([batch, neurons])
    h = tf.zeros([batch, neurons])
    states = tf.tuple([c, h])
    return states

def set_trainable_initial_states(modalities, data_next, batch_size, d_neurons, z_units, saved_initial_states=None):
    out = {m: [] for m in modalities}
    out_initial = {m: [] for m in modalities}
    state = {m: [] for m in modalities}
    z_prior_mean = {m: [] for m in modalities}
    z_prior_var = {m: [] for m in modalities}
    z_prior = {m: [] for m in modalities}
    z_posterior_mean = {m: [] for m in modalities}
    z_posterior_var = {m: [] for m in modalities}
    z_posterior = {m: [] for m in modalities}

    for m in modalities:
        for l in xrange(len(d_neurons[m])): # For each layer
            if l == 0: # Layer 0 has no neurons
                # NB: data is only fed into the network if input_provide_data is True
                if data_next is not None:
                    out[m].append(data_next[m + "_next"][:, 0, :])
                else:
                    out[m].append(tf.zeros([batch_size, d_neurons[m][l]]))
                out_initial[m].append(tf.zeros([batch_size, d_neurons[m][l]]))
                state[m].append(tf.zeros([batch_size, d_neurons[m][l]]))
            else:
                out[m].append(tf.zeros([batch_size, d_neurons[m][l]]))
                out_initial[m].append(tf.zeros([batch_size, d_neurons[m][l]]))
                if saved_initial_states is None:
                    state[m].append(reset_neuron_states(batch_size, d_neurons[m][l])) # (c, h)
                else:
                    state[m].append(saved_initial_states[l])

            z_prior[m].append(tf.ones([batch_size, z_units[m][l]]))
            z_prior_mean[m].append(tf.ones([batch_size, z_units[m][l]]))
            z_prior_var[m].append(tf.ones([batch_size, z_units[m][l]]))
            z_posterior[m].append(tf.ones([batch_size, z_units[m][l]]))
            z_posterior_mean[m].append(tf.ones([batch_size, z_units[m][l]]))
            z_posterior_var[m].append(tf.ones([batch_size, z_units[m][l]]))
    
    print("set_trainable_initial_states: zeroed initial states")

    return internal_states_dict(out=out, out_initial=out_initial, state=state, 
                                z_prior_mean=z_prior_mean, z_prior_var=z_prior_var, z_prior=z_prior,
                                z_posterior_mean=z_posterior_mean, z_posterior_var=z_posterior_var, z_posterior=z_posterior)

# Stability hacks:
eps_minval = 1e-4
eps_maxval = 1e4
sigma_minval = 0.001
sigma_maxval = 10.0
sigma_func = lambda s: s # e.g. tf.tanh

def llog(x):
    return tf.log(tf.clip_by_value(x, eps_minval, eps_maxval))

## Z calculations for Variational Bayes
def calculate_z_prior(idx_layer, prev_out, d_neurons, z_units, batch_size, z_activation_func, scope, limit_sigma=False, override_sigma=None, override_myu=None, override_epsilon=None):
    with tf.variable_scope(scope):
        if override_myu is not None:
            myu = tf.fill([batch_size, z_units[idx_layer]], override_myu)
        else:
            myu = z_activation_func(_linear([prev_out], z_units[idx_layer], bias=True, scope_here='z_prior_myu'))
        if override_sigma is not None:
            s = tf.fill([batch_size, z_units[idx_layer]], override_sigma)
        else:
            s = tf.exp(_linear([prev_out], z_units[idx_layer], bias=True, scope_here='z_prior_sigma'))
        if limit_sigma:
            sigma = tf.clip_by_value(sigma_func(s), sigma_minval, sigma_maxval)
        else:
            sigma = s
        if override_epsilon is not None:
            eps_gaussian = tf.fill([batch_size, z_units[idx_layer]], override_epsilon)
        else:
            eps_gaussian = tf.random_normal([batch_size, z_units[idx_layer]])
        z = tf.add(myu, tf.multiply(sigma, eps_gaussian))
        return z, myu, sigma

# source here is expected to be a learned array (nLayers * nUnits)
# NB: there are two zsrc layers per z layer, first half for myu and second half for sigma
def calculate_z_posterior(idx_layer, prev_out, source, d_neurons, z_units, batch_size, z_activation_func, scope, limit_sigma=False, override_sigma=None, override_myu=None, override_epsilon=None, source_extend=True):
    with tf.variable_scope(scope):
        if source_extend:
            source_extended = _linear([source[:, :z_units[idx_layer]]], z_units[idx_layer]*2, bias=True, scope_here='z_posterior_from_src')
        else:
            source_extended = source[:, :z_units[idx_layer]*2]
        if override_myu is not None:
            myu = tf.fill([batch_size, z_units[idx_layer]], override_myu)
        else:
            if prev_out is not None:
                myu = z_activation_func(_linear([prev_out], z_units[idx_layer], bias=False, scope_here='z_posterior_myu') + source_extended[:, z_units[idx_layer]:])
            else:
                myu = z_activation_func(source_extended[:, z_units[idx_layer]:])
        if override_sigma is not None:
            s = tf.fill([batch_size, z_units[idx_layer]], override_sigma)
        else:
            if prev_out is not None:
                s = tf.exp(_linear([prev_out], z_units[idx_layer], bias=False, scope_here='z_posterior_sigma') + source_extended[:, :z_units[idx_layer]])
            else:
                s = tf.exp(source_extended[:, :z_units[idx_layer]])
        if limit_sigma:
            sigma = tf.clip_by_value(sigma_func(s), sigma_minval, sigma_maxval)
        else:
            sigma = s
        if override_epsilon is not None:
            eps_gaussian = tf.fill([batch_size, z_units[idx_layer]], override_epsilon)
        else:
            eps_gaussian = tf.random_normal([batch_size, z_units[idx_layer]])
        z = tf.add(myu, tf.multiply(sigma, eps_gaussian))
        return z, myu, sigma

# Softmax functions
def softmax(traj, sm_sigma=0.025, sm_minVal=-1.0, sm_maxVal=1.0, softmax_quant=10):
    references = np.linspace(sm_minVal, sm_maxVal, softmax_quant)
    smVal = np.zeros((traj.shape[0],traj.shape[1]*softmax_quant))
    for idxStep in xrange(traj.shape[0]):
        for idxJnt in xrange(traj.shape[1]):
            val = np.zeros((1,softmax_quant))
            sumVal = 0
            for idxRef in xrange(softmax_quant):
                val[0,idxRef] = np.power((references[idxRef] - traj[idxStep,idxJnt]),2)
                val[0,idxRef] = np.exp(-val[0,idxRef] / sm_sigma)
                sumVal = sumVal + val[0,idxRef]
            for idxRef in xrange(softmax_quant):
                val[0,idxRef] = val[0,idxRef] / sumVal
            smVal[idxStep,idxJnt*10:(idxJnt+1)*10] = val[0,:]
    return smVal

def unsoftmax(smVal, sm_minVal=-1.0, sm_maxVal=1.0, softmax_quant=10):
    references = np.linspace(sm_minVal, sm_maxVal, softmax_quant)
    analogJnt = 0
    analog = np.zeros((smVal.shape[0],smVal.shape[1]//softmax_quant))
    for idxJnt in xrange(0,smVal.shape[1],softmax_quant):
        analog[:,analogJnt] = np.matmul(smVal[:,idxJnt:idxJnt+softmax_quant], np.transpose(references))
        analogJnt = analogJnt + 1
    return analog

def tf_unsoftmax(smVal, sm_minVal=-1.0, sm_maxVal=1.0, softmax_quant=10):
    references = tf.convert_to_tensor(np.linspace(sm_minVal, sm_maxVal, softmax_quant), dtype=tf.float32)
    # analogJnt = 0
    # smValShape = np.shape(smVal)
    # analog = list(np.zeros((smValShape[0],smValShape[1]//softmax_quant)))
    analog = []
    for idxJnt in xrange(0,20,softmax_quant):
        analog.append(tf.matmul(smVal[:,idxJnt:idxJnt+softmax_quant], tf.transpose(tf.expand_dims(references,0))))
    return tf.transpose(tf.squeeze(tf.convert_to_tensor(analog, dtype=tf.float32)))

# LeCun (1989)
def extended_hyperbolic(input_matrix):
    return tf.multiply(1.7159, tf.tanh(tf.multiply((tf.divide(2.0, 3.0)), input_matrix)))

def data_mask(motor_data, skip_ahead=None):
    # Input: seq, step, dim
    return tf.sign(tf.reduce_max(tf.abs(motor_data[:, skip_ahead:, :]), 2)) # seq, step

def dropout_mask(dims):
    return tf.cast(tf.random_uniform(dims, dtype=tf.int32, minval=0, maxval=2), dtype=tf.float32)

def windowed_mask(dims, start=[0,1], end=[-1,None], invert=False, end_zeropad=True):
    zeros = tf.zeros(dims, dtype=tf.float32)
    ones = tf.ones(dims, dtype=tf.float32)
    if not invert:
        wmask = tf.concat([ones[:, start[0]:start[1]], zeros[:, start[1]:end[0]], ones[:, end[0]:end[1]]], axis=1)
    else:
        wmask = tf.concat([zeros[:, start[0]:start[1]], ones[:, start[1]:end[0]], zeros[:, end[0]:end[1]]], axis=1)
    if np.shape(wmask)[1] != dims[1]:
        end_pad = dims[1] - np.shape(wmask)[1]
        if end_zeropad:
            wmask = tf.concat([wmask, zeros[:, :end_pad]], axis=1)
        else:
            wmask = tf.concat([wmask, ones[:, :end_pad]], axis=1)
    return wmask

def windowed_dmask(end_dmask, dims, start=[0,1], end=[-1,None], end_zeropad=True): # dims should be [seq, steps, dims]
    zeros = tf.zeros(dims)
    ones = tf.ones(dims)
    goal = tf.cast(tf.expand_dims(tf.reshape(tf.tile(tf.constant(end_dmask), [dims[0]]), [dims[0], dims[2]]), axis=1), tf.float32)
    dmask = tf.concat([ones[:, start[0]:start[1], :], zeros[:, start[1]:end[0], :], goal[:, end[0]:end[1], :]], axis=1)
    if np.shape(dmask)[1] < dims[1]:
        end_pad = dims[1] - np.shape(dmask)[1]
        if end_zeropad:
            dmask = tf.concat([dmask, zeros[:, :end_pad, :]], axis=1)
        else:
            dmask = tf.concat([dmask, ones[:, :end_pad, :]], axis=1)
    return dmask

# For reducing motor trajectories
def td_reduce(src, mask=None, dmask=None):
    if dmask is not None: # apply a granular mask (per channel per timestep)
        src = tf.multiply(src, dmask)
    td = tf.reduce_sum(src, 2) # seq, step
    if mask is not None: # apply a basic mask (per timestep)
        td = tf.multiply(td, mask)

    return tf.reduce_sum(td), td


## Loss functions
# Deprecated windowed_l2_loss
def windowed_l2_loss(xtarget, xprediction, dims, start=[0,1], end=[-1,None], dim_mask=[None,None], get_least_loss=False, return_td=False):
    target = []
    prediction = []
    for n in range(dims[0]):
        target.append(tf_unsoftmax(xtarget[n, :, :], sm_minVal=0.0, sm_maxVal=1.0, softmax_quant=10))
        prediction.append(tf_unsoftmax(xprediction[n, :, :], sm_minVal=0, sm_maxVal=1, softmax_quant=10))

    target = tf.convert_to_tensor(target, dtype=tf.float32)
    prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)

    if not get_least_loss:
        # Starting loss
        loss = tf.nn.l2_loss(target[:,start[0]:start[1],dim_mask[0]:dim_mask[1]] - prediction[:,start[0]:start[1],dim_mask[0]:dim_mask[1]])
        # Ending loss
        loss += tf.nn.l2_loss(target[:,end[0]:end[1],dim_mask[0]:dim_mask[1]] - prediction[:,end[0]:end[1],dim_mask[0]:dim_mask[1]])
        return loss
    else:
        # Starting loss
        start_loss = tf.squared_difference(target[:,start[0]:start[1],dim_mask[0]:dim_mask[1]], prediction[:,start[0]:start[1],dim_mask[0]:dim_mask[1]])
        # Ending loss
        end_loss = tf.squared_difference(target[:,end[0]:end[1],dim_mask[0]:dim_mask[1]], prediction[:,end[0]:end[1],dim_mask[0]:dim_mask[1]])
        if return_td:
            target_shape = tf.shape(target)
            fill_len = target_shape[1] - (tf.shape(start_loss)[1]+tf.shape(end_loss)[1])
            loss = tf.cond(tf.greater(fill_len, 0), lambda: tf.concat([start_loss, tf.zeros([target_shape[0], fill_len, target_shape[2]], dtype=tf.float32), end_loss], axis=1), lambda: tf.concat([start_loss, end_loss], axis=1))
        else:
            loss = tf.concat([start_loss, end_loss], axis=1)
        idx = tf.argmin(loss, 1)

        if return_td:
            return tf.reduce_sum(loss), idx, tf.reduce_sum(tf.gather(loss, idx)), tf.reduce_sum(loss, 2)
        else:
            return tf.reduce_sum(loss), idx, tf.reduce_sum(tf.gather(loss, idx))

def cross_entropy_with_mask(target, prediction, mask=None, dmask=None):
    ce_elem = tf.multiply(tf.negative(target), llog(prediction)) # seq, step, dim*quant_level
    return td_reduce(ce_elem, mask, dmask)

def l2_norm_with_mask(target, output, mask=None, dmask=None):
    diff = tf.squared_difference(target, output) # seq, step, dim*quant_level
    return td_reduce(diff, mask, dmask)

def l1_norm_with_mask(target, output, mask=None, dmask=None):
    diff = tf.abs(target - output)
    return td_reduce(diff, mask, dmask)

def kld_with_mask(target, prediction, mask=None, dmask=None): # seq, step, dim*quant_level
    kl_elem = tf.multiply(target, llog(tf.divide(target, prediction))) # seq, step, dim*quant_level
    return td_reduce(kl_elem, mask, dmask)

def vb_kld(posterior_m, posterior_v, prior_m, prior_v, mask=None, step_multiplier=None, final_divider=None):
    kl = (1.0 + llog(tf.square(posterior_v))
              - llog(tf.square(prior_v))
              + tf.divide((tf.negative(tf.square(posterior_m)) - tf.square(posterior_v) + tf.multiply(2.0, tf.multiply(posterior_m, prior_m)) - tf.square(prior_m)), tf.square(prior_v))
         )
    kl_sum_transpose = tf.transpose(tf.reduce_sum(kl, 2), [1, 0]) #[seq, step]
    if mask is not None:
        kl_sum_transpose = tf.multiply(kl_sum_transpose, mask)
    if step_multiplier is not None:
        kl_sum_tm = tf.multiply(kl_sum_transpose, step_multiplier)
    else:
        kl_sum_tm = kl_sum_transpose

    if final_divider != 1 and final_divider != 0:
        return tf.divide(tf.reduce_sum(kl_sum_tm), final_divider), tf.divide(kl_sum_tm, final_divider)
    else:
        return tf.reduce_sum(kl_sum_tm), kl_sum_tm

# KLD between prior and posterior per layer
def seq_kld_with_mask(source, mask, z_units, modality):
    kld = [0.0]
    # Motor: [step, seq, dim]
    for i in xrange(1, len(z_units)):
        kld.append(vb_kld(source["z_posterior_mean"][modality][i], source["z_posterior_var"][modality][i], source["z_prior_mean"][modality][i], source["z_prior_var"][modality][i], mask, final_divider=z_units[i]))
    return kld

def vb_kld_with_mask(source, mask, z_units, modality, ugaussian_prior=None, ugaussian_prior_by_t=None, ugaussian_weight=None, kld_weight=None, seq_kld_weight_by_t=False):
    kld = [(tf.constant(0.0), tf.constant(0.0))]
    # Motor: [step, seq, dim]
    for i in xrange(1, len(z_units)):
        if not seq_kld_weight_by_t:
            step_multiplier = None
        elif ugaussian_prior_by_t is None:
            pshape = tf.shape(source["z_prior_mean"][modality][i])
            step_multiplier = tf.fill([pshape[0]], kld_weight[i-1])

        qm = source["z_posterior_mean"][modality][i]
        qv = source["z_posterior_var"][modality][i]
        if ugaussian_prior is not None and ugaussian_prior[i] == False:
            pshape = tf.shape(source["z_prior_mean"][modality][i])
            pm = tf.fill(pshape, 0.0)
            pv = tf.fill(pshape, 1.0)
        elif ugaussian_prior_by_t is not None:
            pshape = tf.shape(source["z_prior_mean"][modality][i])
            pm = tf.concat([source["z_prior_mean"][modality][i][:ugaussian_prior_by_t[0], :, :], tf.fill([ugaussian_prior_by_t[1]-ugaussian_prior_by_t[0], pshape[1], pshape[2]], 0.0), source["z_prior_mean"][modality][i][ugaussian_prior_by_t[1]:, :, :]], axis=0)
            pv = tf.concat([source["z_prior_var"][modality][i][:ugaussian_prior_by_t[0], :, :], tf.fill([ugaussian_prior_by_t[1]-ugaussian_prior_by_t[0], pshape[1], pshape[2]], 1.0), source["z_prior_var"][modality][i][ugaussian_prior_by_t[1]:, :, :]], axis=0)
            step_multiplier = tf.concat([[tf.constant(ugaussian_weight)], tf.fill([pshape[0]-1], kld_weight[i-1])], axis=0)
        else:
            pm = source["z_prior_mean"][modality][i]
            pv = source["z_prior_var"][modality][i]
        kld.append(vb_kld(qm, qv, pm, pv, mask, step_multiplier, final_divider=z_units[i]))
    return kld

## State dictionary must be consistent
def internal_states_dict(t_step=0, out=None, out_initial=None, state=None, 
                         z_prior_mean=None, z_prior_var=None, z_prior=None,
                         z_posterior_mean=None, z_posterior_var=None, z_posterior=None):
    states = dict()
    states['t_step'] = t_step
    states['out'] = out
    states['out_initial'] = out_initial
    states['state'] = state
    states['z_prior_mean'] = z_prior_mean
    states['z_prior_var'] = z_prior_var
    states['z_prior'] = z_prior
    states['z_posterior_mean'] = z_posterior_mean
    states['z_posterior_var'] = z_posterior_var
    states['z_posterior'] = z_posterior

    # Debugging
    # import inspect
    # import pprint
    # print("internal_states_dict (caller: %s):" % (inspect.getouterframes(inspect.currentframe(), 2)[1][3]))
    # pp = pprint.PrettyPrinter()
    # pp.pprint(out)
    return states

def update_internal_states_dict(previous_states, t_step=None, out=None, out_initial=None, state=None, 
                                z_prior_mean=None, z_prior_var=None, z_prior=None,
                                z_posterior_mean=None, z_posterior_var=None, z_posterior=None):
    states = previous_states
    if t_step is not None:
        states['t_step'] = t_step
    if out is not None:
        states['out'] = out
    if out_initial is not None:
        states['out_initial'] = out_initial
    if state is not None:
        states['state'] = state
    if z_prior_mean is not None:
        states['z_prior_mean'] = z_prior_mean
    if z_prior_var is not None:
        states['z_prior_var'] = z_prior_var
    if z_prior is not None:
        states['z_prior'] = z_prior
    if z_posterior_mean is not None:
        states['z_posterior_mean'] = z_posterior_mean
    if z_posterior_var is not None:
        states['z_posterior_var'] = z_posterior_var
    if z_posterior is not None:
        states['z_posterior'] = z_posterior

    # Debugging
    # import inspect
    # import pprint
    # print("update_internal_states_dict (caller: %s):" % (inspect.getouterframes(inspect.currentframe(), 2)[1][3]))
    # pp = pprint.PrettyPrinter()
    # pp.pprint(out)
    return states
