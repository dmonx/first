# Train the network and save the weights after training
model.fit(train_x, train_y, batch_size=20, epochs=10000, verbose=1)
model.save_weights('weights.h5')