import tensorflow as tf
import numpy as np
import logging as log
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split

class LR_tf:
    def __init__(self, k = 1, learningRate = 0.003, batchSize = 30, iterNum = 1500, logLevel = log.WARNING):
        self.k = tf.constant(k, dtype=tf.float32)
        self.sess = tf.Session()

        # Define the learning rate， batch_size etc.
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.iterNum = iterNum

        log.basicConfig(level=logLevel, format='%(asctime)s %(levelname)s %(message)s')

    def __del__(self):
        self.sess.close()

    def fit(self, X, y):
        # Begin building the model framework
        # Declare the variables that need to be learned and initialization
        # There are 4 features here, A's dimension is (4, 1)
        A = tf.Variable(tf.random_normal(shape=[X.shape[1], 1]))
        b = tf.Variable(tf.random_normal(shape=[1, 1]))
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Define placeholders
        data = tf.placeholder(dtype=tf.float32, shape=[None, X.shape[1]])
        target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Declare the model you need to learn
        z = tf.matmul(data, A) + b
        u = tf.pow(tf.sqrt(3.0) * tf.sqrt(27.0 * tf.pow(z, 2.0) + 4.0 * tf.pow(self.k, 3.0)) - 9.0 * z, 1 / 3)
        elem1 = tf.pow(2 / 3, 1 / 3) * self.k / u
        elem2 = u / (tf.pow(2.0, 1 / 3) * tf.pow(3.0, 2 / 3))
        zTrans = elem1 - elem2


        # Declare loss function
        # Use the sigmoid cross-entropy loss function,
        # first doing a sigmoid on the model result and then using the cross-entropy loss function
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=zTrans, labels=target))

        # Define the optimizer
        opt = tf.train.GradientDescentOptimizer(self.learningRate)

        # Define the goal
        goal = opt.minimize(loss)

        # Define the accuracy
        # The default threshold is 0.5, rounded off directly
        prediction = tf.round(tf.sigmoid(zTrans))
        # Bool into float32 type
        correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
        # Average
        accuracy = tf.reduce_mean(correct)
        # End of the definition of the model framework

        # Start training model
        # Define the variable that stores the result
        loss_trace = []
        train_acc = []

        # training model
        for epoch in range(self.iterNum):
            # Generate random batch index
            batch_index = np.random.choice(len(X), size=self.batchSize)
            batch_train_X = X[batch_index]
            batch_train_y = np.matrix(y[batch_index]).T
            self.sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})
            temp_loss = self.sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
            # convert into a matrix, and the shape of the placeholder to correspond
            temp_train_acc = self.sess.run(accuracy, feed_dict={data: X, target: np.matrix(y).T})
            # recode the result
            loss_trace.append(temp_loss)
            train_acc.append(temp_train_acc)
            # output
            if (epoch + 1) % 300 == 0:
                print('epoch: {:4d} loss: {:5f} train_acc: {:5f}'.format(epoch + 1, temp_loss, temp_train_acc))

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def get_params(self, deep=True):
        pass


#############################################################
# X, y = load_breast_cancer(return_X_y=True)
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

#############################################################