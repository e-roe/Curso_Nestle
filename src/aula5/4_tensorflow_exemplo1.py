import tensorflow as tf

# Cria tensores
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])

# Soma os tensores
z = tf.add(x, y)
print(z.numpy())
