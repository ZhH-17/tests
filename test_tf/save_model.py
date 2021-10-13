import tensorflow as tf
import tensorflow.keras.backend
import keras
import keras.backend as K
import numpy as np
import datetime
import os
import pdb


def save_model_to_serving(model, export_version, export_path='prod_models'):
    print("input: ", model.input)
    print("output: ", model.output)
    pdb.set_trace()
    # inputs = {t.name: t for t in model.input}
    # outputs = {t.name: t for t in model.output}
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={"input_1:0": model.input},
        outputs={"dense2:BiasAdd:0": model.output})
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(), tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={'detection': signature,},
        legacy_init_op=legacy_init_op)
    builder.save()


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('savepath',
                        metavar="/path/to/save",
                        help="Path to save model")
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--version', '-v', required=False,
                        default=1, help="model version")

    args = parser.parse_args()
    model_path = args.model
    model_save_path = args.savepath

    inputs = keras.layers.Input(shape=(4,))
    x = keras.layers.Dense(20, activation='relu', name="dense1")(inputs)
    outputs = keras.layers.Dense(3, name="dense2")(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    # model = tf.keras.Sequential([
    #     KL.Dense(20, activation='relu', name="dense1"),
    #     KL.Dense(3, name="dense2")])

    model.load_weights(model_path, by_name=True)
    save_model_to_serving(model, args.version, model_save_path)

