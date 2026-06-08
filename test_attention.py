import tensorflow as tf
from attention import AttentionLayer

x = tf.random.normal((32, 100, 128))

attention = AttentionLayer()

output = attention(x)

print(output.shape)