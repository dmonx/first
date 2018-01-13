TARGET_DIM = 16

# Encoder
inputs = Input(shape=(784,))
h_encode = Dense(256, activation='relu')(inputs)
h_encode = Dense(128, activation='relu')(h_encode)
h_encode = Dense(64, activation='relu')(h_encode)
h_encode = Dense(32, activation='relu')(h_encode)

# Coded
encoded = Dense(TARGET_DIM, activation='relu')(h_encode)

# Decoder
h_decode = Dense(32, activation='relu')(encoded)
h_decode = Dense(64, activation='relu')(h_decode)
h_decode = Dense(128, activation='relu')(h_decode)
h_decode = Dense(256, activation='relu')(h_decode)
outputs = Dense(784, activation='linear')(h_decode)

# Autoencoder Model
autoencoder = Model(inputs=inputs, outputs=outputs)

# Encoder Model
encoder = Model(inputs=inputs, outputs=encoded)

# Optimizer / Update Rule
adam = Adam(lr=0.001)

# Compile the model Binary Crossentropy
autoencoder.compile(optimizer=adam, loss='mse')

autoencoder.fit(train_x, train_x, batch_size=256, epochs=100, verbose=1, shuffle=True, validation_data=(test_x, test_x))
autoencoder.save_weights('weightsLinear.h5')