# Download and Save MNIST Dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Scale from 0 to 1
train_x = train_x.astype('float32') / 255.
test_x = test_x.astype('float32') / 255.

# Reshape from 28x28 matrix to 784 vector
train_x = np.reshape(train_x, (len(train_x), np.prod(train_x.shape[1:])))
test_x = np.reshape(test_x, (len(test_x), np.prod(test_x.shape[1:])))