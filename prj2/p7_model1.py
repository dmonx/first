# Feature Extraction Layer
inputs = Input(shape=(28, 28, 1))
conv_layer = ZeroPadding2D(padding=(2,2))(inputs) 
conv_layer = Conv2D(16, (5, 5), strides=(3,3), activation='relu')(conv_layer) 
conv_layer = MaxPooling2D((2, 2))(conv_layer) 
conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer) 
conv_layer = ZeroPadding2D(padding=(1,1))(conv_layer) 
conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer)

# Flatten feature map to Vector with 576 element.
flatten = Flatten()(conv_layer) 

# Fully Connected Layer
fc_layer = Dense(256, activation='relu')(flatten)
fc_layer = Dense(64, activation='relu')(fc_layer)
outputs = Dense(10, activation='softmax')(fc_layer)

model = Model(inputs=inputs, outputs=outputs)

# Adam Optimizer and Cross Entropy Loss
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Print Model Summary
print(model.summary())