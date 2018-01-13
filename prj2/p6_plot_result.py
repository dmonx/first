# Encoded Data
encoded_train = encoder.predict(train_x)
encoded_test = encoder.predict(test_x)

# Reconstructed Data
reconstructed = autoencoder.predict(test_x)

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
	count = 0
	while True:
		if i == test_y[count]:
			# Original
			ax = plt.subplot(2, n, i + 1)
			plt.imshow(test_x[count].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# Reconstructed
			ax = plt.subplot(2, n, i + 1 + n)
			plt.imshow(reconstructed[count].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			break;

		count += 1
plt.show()