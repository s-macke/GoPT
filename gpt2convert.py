import sys
import tensorflow as tf
import numpy as np
import struct

def load_tf_weights(tf_path, out_filename):
    print("tf_path=" + tf_path)
    f = open(out_filename, "wb")
    init_vars = tf.train.list_variables(tf_path)
    for name, shape in init_vars:
        out_name = name
        if name[0:6] == "model/":
            out_name = name[6:]
        print("name", out_name, "shape",shape)
        array = tf.train.load_variable(tf_path, name)
        array = array.squeeze()
        n_dims = len(array.shape)

        # convert to float16
        type_id = 0
        #if out_name == "wte" or out_name.find("/w") >= 0:
        #    type_id = 2
        #    array = array.astype(np.float16)
        # variable header
        str = out_name.encode('utf-8')
        f.write(struct.pack("iiii", 0x23f4aefa, type_id, n_dims, len(str)))
        for i in range(n_dims):
            f.write(struct.pack("i", array.shape[n_dims - 1 - i]))
        f.write(str);
        # variable data
        array.tofile(f)
    f.close()

if len(sys.argv) != 3:
    print("""usage: gpt2convert.py gpt2_model_dir output_file

Convert a Tensorflow GPT-2 parameter dump to a LibNC one.

gpt2_model_dir is the directory containing the Tensorflow parameter dump
output_file is filename of the libNC parameter dump""")
    sys.exit(1)

load_tf_weights(sys.argv[1], sys.argv[2])
