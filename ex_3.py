from time import time
import numpy as np


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    x -= np.max(x)  # For numeric stability
    x = np.exp(x)
    x /= np.sum(x)

    return x


def sigmoid(x):
    """
    Compute the sigmoid vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of sigmoid values.
    """
    return np.array([1 / (1 + np.exp(-i)) for i in x])


class MLP1(object):
    def __init__(self, in_dim, hid_dim, out_dim):
        # Xavier Glorot init
        Glorot_init = lambda n, m: np.random.uniform(-np.sqrt(6.0 / (n + m)), np.sqrt(6.0 / (n + m)),
                                                     (n, m) if (n != 1 and m != 1) else n * m)

        self.U = Glorot_init(hid_dim, out_dim)
        self.W = Glorot_init(in_dim, hid_dim)
        self.b = Glorot_init(1, hid_dim)
        self.b_tag = Glorot_init(1, out_dim)

    def forward(self, x):
        """
        :param x: numpy array of size in_dim.
        :return: numpy array of size out_dim.
        """
        sig = sigmoid(np.dot(x, self.W) + self.b)
        return softmax(np.dot(sig, self.U) + self.b_tag)

    def predict_on(self, x):
        """
        :param x: numpy array of size in_dim.
        :return: scalar to indicate the predicted label of x.
        """
        return np.argmax(self.forward(x))

    def loss_and_gradients(self, x, y):
        """
        :param x: numpy array of size in_dim.
        :param y: scalar, label of x.
        :return: loss (float) and gradients (list of size 4).
        """
        sig = sigmoid(np.dot(x, self.W) + self.b)
        y_hat = softmax(np.dot(sig, self.U) + self.b_tag)
        loss = -np.log(y_hat[y])  # NLL loss

        layer1_der = sig * (1 - sig)

        # gradient of b_tag
        gb_tag = np.copy(y_hat)
        gb_tag[y] -= 1

        # gradient of U
        gU = np.outer(sig, y_hat)
        gU[:, y] -= sig

        # gradient of b - use the chain rule
        dloss_dsigmoid = -self.U[:, y] + np.dot(self.U, y_hat)
        dsigmoid_db = layer1_der
        gb = dloss_dsigmoid * dsigmoid_db

        # gradient of W - use the chain rule
        gW = np.outer(x, gb)

        return loss, [gU, gW, gb, gb_tag]

    def accuracy_on_dataset(self, dataset):
        """
        :param dataset: list of tuples, each is (x, y) where x is vector and y is its label.
        :return: accuracy of the model on the given dataset, float between 0 to 1.
        """
        good = 0.0
        for x, y in dataset:
            y_prediction = self.predict_on(x)
            if y_prediction == y:
                good += 1
        return good / len(dataset)

    def get_params(self):
        """
        :return: list of model parameters.
        """
        return [self.U, self.W, self.b, self.b_tag]

    def set_params(self, params):
        """
        :param params: list of size 4.
        """
        self.U = params[0]
        self.W = params[1]
        self.b = params[2]
        self.b_tag = params[3]


def train_classifier(train_data, dev_data, model,
                     num_epochs=30, learning_rate=0.01, batch_size=8):
    """
    train the model on the given train-set, evaluate its performance on dev-set.
    after training, the best parameters of the model will be set to the model.
    :param train_data: array-like of tuples, each is (x, y) where x is numpy array and y is its label.
    :param dev_data: array-like of tuples, each is (x, y) where x is numpy array and y is its label.
    :param model: NN model.
    :param num_epochs: number of epochs.
    :param learning_rate: float.
    :param batch_size: size of batch.
    """
    best_params = [np.copy(param) for param in model.get_params()]
    best_acc = 0.0

    U_shape = model.U.shape
    W_shape = model.W.shape
    b_shape = model.b.shape
    b_tag_shape = model.b_tag.shape

    def zero_grads_for_batch():
        gU = np.zeros(U_shape)
        gW = np.zeros(W_shape)
        gb = np.zeros(b_shape[0])
        gb_tag = np.zeros(b_tag_shape[0])
        return [gU, gW, gb, gb_tag]

    batch_size_modulo = batch_size - 1

    for epoch in xrange(num_epochs):
        t_epoch = time()
        total_loss = 0.0  # total loss in this iteration.
        np.random.shuffle(train_data)

        batch_loss = 0
        batch_grads = zero_grads_for_batch()
        for i, (x, y) in enumerate(train_data):
            loss, grads = model.loss_and_gradients(x, y)

            batch_loss += loss
            batch_grads[0] += grads[0]  # U
            batch_grads[1] += grads[1]  # W
            batch_grads[2] += grads[2]  # b
            batch_grads[3] += grads[3]  # b tag

            if i % batch_size == batch_size_modulo:  # SGD update parameters
                model.U -= learning_rate * batch_grads[0]  # U update
                model.W -= learning_rate * batch_grads[1]  # W update
                model.b -= learning_rate * batch_grads[2]  # b update
                model.b_tag -= learning_rate * batch_grads[3]  # b_tag update

                total_loss += batch_loss
                batch_loss = 0
                batch_grads = zero_grads_for_batch()

        if batch_loss != 0:  # there are leftovers from the data that is not in size of batch
            # SGD update parameters
            model.U -= learning_rate * batch_grads[0]  # U update
            model.W -= learning_rate * batch_grads[1]  # W update
            model.b -= learning_rate * batch_grads[2]  # b update
            model.b_tag -= learning_rate * batch_grads[3]  # b_tag update

            total_loss += batch_loss

        # notify progress
        train_loss = total_loss / len(train_data)
        dev_accuracy = model.accuracy_on_dataset(dev_data)
        if dev_accuracy > best_acc:
            best_params = [np.copy(param) for param in model.get_params()]
            best_acc = dev_accuracy
        print epoch, 'train_loss:', train_loss, 'time:', time() - t_epoch, 'dev_acc:', dev_accuracy

    print 'best accuracy:', best_acc
    model.set_params(best_params)


def train_dev_split(train_x, train_y, size=0.2):
    """
    :param train_x: numpy array of vectors.
    :param train_y: numpy array of integers, each is label associated with train_x.
    :param size: percentage of how much to take from the train data to become dev data.
    """
    train_data = zip(train_x, train_y)
    np.random.shuffle(train_data)
    size = int(len(train_data) * size)

    dev_data = train_data[:size]
    train_data = train_data[size:]
    return train_data, dev_data


def predict_test(test_x, model):
    """
    create a file which contains the prediction of the model on a blind test.
    :param test_x: numpy array of vectors.
    :param model: NN model.
    """
    with open('test.pred', 'w') as f:
        def predict_as_str(x):
            return str(model.predict_on(x))

        preds = map(predict_as_str, test_x)
        f.write('\n'.join(preds))


def main():
    start = time()

    # load data
    print 'loading data'
    train_x = np.loadtxt('train_x') / 255  # normalized
    train_y = np.loadtxt('train_y', dtype=int)

    print 'time to load data:', time() - start
    start = time()

    # set dims
    in_dim = train_x[0].shape[0]
    hid_dim = 256
    out_dim = 10

    # create and train classifier
    print 'start training'
    model = MLP1(in_dim, hid_dim, out_dim)
    train_data, dev_data = train_dev_split(train_x, train_y, size=0.2)
    print 'all:', len(train_x), ', train:', len(train_data), 'dev:', len(dev_data)
    train_classifier(train_data, dev_data, model)

    # blind test
    print 'start blind test'
    test_x = np.loadtxt('test_x') / 255  # normalized
    predict_test(test_x, model)

    print 'time to train:', time() - start


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0
