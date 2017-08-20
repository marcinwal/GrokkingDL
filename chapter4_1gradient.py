weight = 0.5
input = 0.5
goal_prediction =0.8

step_amount = 0.001

for iteration in range(1101):
    
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2
    
    print('Error:{} Prediction: {}'.format(error, prediction))
    
    up_prediction = input * (weight + step_amount)
    up_error = (goal_prediction - up_prediction) ** 2 
    
    down_preciction = input * (weight - step_amount)
    down_error = (goal_prediction - down_preciction) ** 2 
    
    if (down_error < up_error):
        weight -= step_amount
    else:
        weight += step_amount