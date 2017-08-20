input = 2
goal_pred = 0.8
weight = 0.6
alpha = 0.1

for interation in range(20):
	prediction = input * weight
	error = (prediction - goal_pred) ** 2
	delta = prediction - goal_pred
	weight_delta = delta * input
	weight -= weight_delta * alpha
	
	print('Error:{} Prediction: {}'.format(error, prediction))