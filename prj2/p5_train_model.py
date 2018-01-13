# Train the model and use validation data
model.fit(train_x, train_y, batch_size=16, epochs=5000, verbose=1, validation_data=(val_x, val_y))
model.save_weights('weights.h5')

# Predict all Validation data
predict = model.predict(val_x)

# Visualize Prediction
df = pd.DataFrame(predict)
df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
df.index = val_data[:,0]
print(df)