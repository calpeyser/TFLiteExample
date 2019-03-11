import tensorflow as tf

# A very simple trainable model: y = sin(x) + offset, where
# offset is trainable.

offset = tf.get_variable("offset", [1,], tf.float32)
x = tf.placeholder(tf.float32, shape=(None,))
y = tf.sin(x + offset)
y_ = tf.placeholder(tf.float32, shape=(None,))
loss = tf.reduce_sum(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()


# Training data that implies offset ~= 0.4
TRAINING_DATA = [
	[[0.4], [0.84]],
	[[1.4], [1.32]],
	[[-0.7], [-0.26]],
]

with tf.Session() as sess:
	sess.run(init_op)
	for i in range(100):
		ex = TRAINING_DATA[i % 3]
		sess.run(train, feed_dict={x: ex[0], y_: ex[1]})

	offset = sess.run(offset)
	print(offset)
	tf.saved_model.simple_save(sess,
	            "/tmp/models/",
	            inputs={"x": x},
	            outputs={"y": y})

