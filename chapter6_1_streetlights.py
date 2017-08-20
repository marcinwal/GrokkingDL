import numpy as np 
#example with street lights


weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1
#                         R Y G      
streetlights = np.array([[1,0,1],
                         [0,1,1],
                         [1,1,1],
                         [0,1,1],
                         [1,0,1]])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

input = streetlights[0]
goal_prediction = walk_vs_stop[0] # stop

for iteration in range(20):
    prediction = input.dot(weights)
    error = (prediction - goal_prediction) ** 2
    delta = prediction - goal_prediction
    weighted_delta = delta * input
    weights -= alpha * weighted_delta
    print('error: {} prediction: {}'.format(error,prediction))