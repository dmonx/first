# Use TensorBoard
callbacks = TensorBoard(log_dir='./Graph')

# Train for 100 Epochs and use TensorBoard Callback
model.fit(train_x, train_y, batch_size=256, epochs=100, verbose=1, validation_data=(test_x, test_y), callbacks=[callbacks])

# Save Weights
model.save_weights('weights.h5')