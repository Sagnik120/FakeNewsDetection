import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(
            tf.tensordot(inputs, self.W, axes=1) + self.b
        )

        weights = tf.nn.softmax(score, axis=1)

        context = weights * inputs
        context = tf.reduce_sum(context, axis=1)

        return context