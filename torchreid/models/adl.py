#import tensorflow as tf
import numpy as np
import torch

#def gating_op(input_, option):
#    if option.method_name == 'CAM':
#        output = input_
#    elif option.method_name == 'ADL':
#        output = attention_based_dropout(input_, option)
#    else:
#       raise KeyError("Unavailable method: {}".format(option.method_name))

#    return output


def attention_based_dropout(input_):
    def _get_importance_map(attention):
        return torch.sigmoid(attention)

    def _get_drop_mask(attention, drop_thr):
        max_val = torch.max(attention) #max_val = torch.max(attention, axis=[1, 2, 3], keepdims=True)
        thr_val = max_val * drop_thr
        return (attention < thr_val).float()

    def _select_component(importance_map, drop_mask, drop_prob):
        random_tensor = torch.FloatTensor(1).uniform_(drop_prob,1. + drop_prob)[0]   # random_tensor = tf.random_uniform([], drop_prob, 1. + drop_prob)
        binary_tensor = torch.floor(random_tensor).float()  #  binary_tensor = tf.cast(tf.floor(random_tensor), dtype=tf.float32)
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    
    adl_keep_prob = 0.90
    adl_threshold = 0.70
    drop_prob = 1 - adl_keep_prob
    drop_thr = adl_threshold

 #   if training:
    attention_map = torch.mean(input_, axis=1, keepdims=True)
    importance_map = _get_importance_map(attention_map)
    drop_mask = _get_drop_mask(attention_map, drop_thr)
    selected_map = _select_component(importance_map, drop_mask, drop_prob)
    output = input_ * selected_map
    return output

#    else:
#        return input_


