import numpy as np


#hidden layer added
# toes %wins fans
ih_wgt = np.array([
    [0.1, 0.2, -0.1], #hid[0]
    [-0.1, 0.1, 0.9], #hid[1]
    [0.1, 0.4, 0.1]]).T #hid[2]
    
hp_wgt = np.array([
    [0.3, 1.1, -0.3], #hurt
    [0.1, 0.2, 0.0], #win
    [0.0, 1.3, 0.1]]).T #sad 

weights = [ih_wgt, hp_wgt]

def neural_network(input, weights):
    hid = input.dot(weights[0])
    print(hid)
    pred = hid.dot(weights[1])
    return pred

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

rand_test = np.random.rand(3,4)
print(rand_test)
print(rand_test.shape)

input = np.array([toes[0], wlrec[0], nfans[0]])

# print(input)
# print(weights)


pred = neural_network(input, weights)

print(pred)