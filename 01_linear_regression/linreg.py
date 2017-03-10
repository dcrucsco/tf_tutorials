import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_diabetes, load_boston
from sklearn.model_selection import train_test_split

def get_data():
    # Replace the below function call to apply linear regression on different dataset
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    return features, labels

def normalize(dataset):
	# Check details here https://en.wikipedia.org/wiki/Feature_scaling
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def add_bias_reshape(features,labels):
	# Add b term of the equation Y = W * X + b, which will be all 1's
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l


features,labels = get_data()
normalized_features = normalize(features)
f, l = add_bias_reshape(normalized_features, labels)
n_dim = f.shape[1]

X_train, x_test, Y_train, y_test = train_test_split(f, l, test_size=0.1, random_state=42)

learning_rate = 0.01
training_epochs = 1000
loss_history = np.empty(shape=[1], dtype=float)

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.ones([n_dim, 1]))


init = tf.global_variables_initializer()

y_ = tf.matmul(X,W)

loss = tf.reduce_mean(tf.square(y_-Y))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		sess.run(train_step, feed_dict={X:X_train, Y:Y_train})
		loss_history = np.append(loss_history, sess.run(loss, feed_dict={X:X_train, Y:Y_train}))
	y_pred = sess.run(y_, feed_dict={X:x_test})

	mse = tf.reduce_mean(tf.square(y_pred - y_test))
	print sess.run(mse)

# Plot for Measure vs Predicted target value
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# Plot for loss decay over epochs
# plt.plot(range(len(loss_history)), loss_history)
# plt.axis([0, training_epochs, 0, np.max(loss_history)])
# plt.show()
