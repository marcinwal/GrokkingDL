# regularization as the way to avoid overfitting


#ADDING DROPOUT to avoid ofverfitting

import numpy as np
import syslog as sys

np.random.seed(1)


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output >= 0


alpha = 0.005
iterations = 300
hidden_size = 400

pixels_per_image = 784
num_labels = 10

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in xrange(iterations):

    error = 0.0
    correct_cnt = 0

    for i in xrange(len(images)):
        layer_0 = images[i:i + 1]

        dropout_mask = np.random.randint(2, size=layer_1.shape)   #DROPOUT

        layer_1 = relu(np.dot(layer_0, weights_0_1))

        layer_1 *= dropout_mask * 2                                 #DROPOUT


        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((labels[i:i + 1] - layer_2) ** 2)

        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i + 1]))

        layer_2_delta = (labels[i:i + 1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        layer_1_delta *= dropout_mask                               #DROPOUT

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        sys.stdout.write("\r" + "I:" + str(j) + "Error:" + str(error / float(len(images)))[0:5]) + "Correct:" + str(correct_cnt / float(len(images)))

error = 0.0
correct_cnt = 0

for i in xrange(len(test_images)):
    layer_0 = test_images[i:i + 1]
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = np.dot(layer_1, weights_1_2)

    error += np.sum(test_labels[i:i + 1] - layer_2) ** 2)
    correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

    sys.stdout.write("\r" + "I:" + str(j) + "Error:" + str(error / float(len(test_images)))[0:5]) + \
    + "Correct:" + str(correct_cnt / float(len(test_images)))