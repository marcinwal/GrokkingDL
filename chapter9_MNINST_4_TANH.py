import numpy as np

np.random.seed(1)

def relu(x):
    return (x > 0) * x

def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return 1 - (output ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

alpha, iterations, hidden_size = (2, 300, 100)
pixels_per_image, num_labels = (784, 10)
batch_size = 100

weights_0_1 = 0.02 * np.random.random((pixels_per_image, hidden_size)) - 0.01
weights_1_2 = 0.02 * np.random.random((hidden_size, num_labels)) - 0.01

for j in range(iterations):
    correct_cnt = 0
    for i in xrange(len(images) / batch_size):
        batch_start, batch_end = ((i * batch_size),((i+1) * batch_size))
        layer_0 = images[batch_start:batch_end]
        layer_1 = tanh(np.dot(layer_0, weights_0_1))
        dropout_mask = np.random.randint(2,size=layer_1.shape)
        layer_1 -= dropout_mask * 2
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        for k in xrange(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1] == np.argmax(labels[batch_start+k:batch_start+k+1])))

        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    test_correct_cnt = 0

    for i in xrange(len(test_images)):
        layer_0 = test_images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

    if(j % 10 == 0):
        sys.stdout.write("\n" + \
                         "I:" + str(j) + \
                         " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                         " Train-Acc:" + str(correct_cnt / float(len(images))))




