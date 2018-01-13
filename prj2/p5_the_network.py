# Create Network
inputs = Input(shape=(16,))
h_layer = Dense(10, activation='sigmoid')(inputs)

# Softmax Activation for Multiclass Classification
outputs = Dense(3, activation='softmax')(h_layer)

model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)

# Compile the model with Cross Entropy Loss
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])