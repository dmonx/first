def forwardPass(inputs, weight, bias):
	w_sum = np.dot(inputs, weight) + bias

	# Linear Activation f(x) = x
	act = w_sum

	return act