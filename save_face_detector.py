import sys
import time
import numpy as np
import tensorflow as tf

PATH_TO_CKPT = './frozen_inference_graph_face.pb'
export_path = './out'

def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    return (inputs, outputs)

with tf.Graph().as_default() as detection_graph:
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.compat.v1.Session(graph=detection_graph, config=config)

  (inputs, outputs) = analyze_inputs_outputs(detection_graph)

  print("save!")
  tf.saved_model.save(od_graph_def, export_path)

"""
tf.saved_model.simple_save(
  sess,
  export_path,
  inputs={'inputs': inputs},
  outputs={'outputs': outputs})
"""