# Load your data set
import numpy as np
import math
x_train = np.array([2.1, 3.5, 1.2, 5.8, 0.9, 4.7, 2.8, 6.3, 0.5, 3.9])   #known O.D
y_train = np.array([4.2, 6.7, 2.4, 7.1, 3.3, 8.5, 5.0, 9.2, 2.0, 6.0])   #known concentration
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

a , b  = compute_gradient(x_train,y_train,0,0)
print(a,b)

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
 
    return w, b, J_history, p_history #return w and J,w history for graphing
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 100000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"{w_final:.4f},{b_final:.4f}")
import matplotlib.pyplot as plt

plt.scatter(x_train, y_train, label="Data Points", color="blue")

# Define the slope and intercept of the line
m = w_final
b = b_final

# Create a range of x-values for the line
x_line = np.linspace(0, 6, 100)

# Calculate y-values for the line
y_line = m * x_line + b

# Plot the line without connecting it to the data points
plt.plot(x_line, y_line, label=f"Line y = {w_final}x + {b_final}", color="red", linestyle='--')

# Customize the plot
plt.xlabel("O.D")
plt.ylabel("CONCENTRATION")
plt.title("Concentration Prediction Line")
plt.legend()

# Show the plot
plt.show()

while True:   
    try:
        unknown = float(input('what\'s the unknown\'s O.D\n to exit promt "exit":'))
        print(f"the predicted concentration is {(w_final*unknown)+b_final}")  
    except  ValueError:
        print("thankyou for using")
        break

    



   
    
        
