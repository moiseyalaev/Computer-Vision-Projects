#%%
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# %%
# errors loading in data so I used following code snippit
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# %%
# Show the training data
plt.imshow(X_train[0])
plt.show()

# %%
# Normalize data so vals are in [0,1] using python vectorized division
X_train /= 255.0
X_test /=  255.0

# %%
# Build simple computer vision model as base model
# Always flatten non-one-dimensional data, usually specify data input_shape for clarity
CV_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                       tf.keras.layers.Dense(128, activation='relu'),
                                       tf.keras.layers.Dense(10, activation='softmax')])

# %%
# Train model on training set: Compile and fit the model
# Use sparse categorical c.e since targets are one hot encoded
CV_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

CV_model.fit(X_train, Y_train, epochs=5)    # loss: 0.5370 - accuracy: 0.8164
# %%
# Evaluate model on test data
CV_model.evaluate(X_test, Y_test)           # loss: 0.5740 - accuracy: 0.8046

# %%
# Make predictions of the classes given a test images
classifications = CV_model.predict(X_test)
print(classifications[0]) # probability in last position highest so it matches with label of 9
print(Y_test[0]) #

# Classifications := a 2D array with each row representing the
# probability of an img being one of the 10 available classes

# %%
# Create better new model with more neurons in first hidden layer
CV_model1 = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                       tf.keras.layers.Dense(512, activation='relu'),
                                       tf.keras.layers.Dense(10, activation='softmax')])

CV_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
CV_model1.fit(X_train, Y_train, epochs=5)   # neurons=512  => loss: 0.4968 - accuracy: 0.8309;
                                            # neurons=1024 => loss: 0.4823 - accuracy: 0.8407

CV_model1.evaluate(X_test, Y_test)          # neurons=512  => loss: 0.6322 - accuracy: 0.7907
                                            # neurons=1012 => loss: 0.7241 - accuracy: 0.8208
                                            # NOTE: 512 is a good amount, 1012 is overfitting and not worth computation

classifications1 = CV_model1.predict(X_test)
print(classifications1[0])
print(Y_test[0])
# %%
# Build a third model that is deeper (more layers)
CV_model2 = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

CV_model2.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', metrics=["accuracy"])
CV_model2.fit(X_train, Y_train, epochs=5)       # neurons=512,256  => loss: 0.3924 - accuracy: 0.8599
                                            # neurons=1024,512 => loss: 0.3913 - accuracy: 0.8619

CV_model2.evaluate(X_test, Y_test)              # neurons=512,256  => loss: 0.4559 - accuracy: 0.8430
                                            # neurons=1012.512 => loss: 0.4613 - accuracy: 0.8375
                                            # NOTE: 1012,512 is likely overfitting and not worth the dimensions
classifications = CV_model2.predict(X_test)
print(classifications[0])
print(Y_test[0])

# %%
# Build a third model that has more epochs and combines the conclusions of prev two models
CV_model3 = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

CV_model3.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', metrics=["accuracy"])
CV_model3.fit(X_train, Y_train, epochs=10)          # epochs=10, neurons=512 => loss: 0.4499 - accuracy: 0.8469
                                                    # epochs=10 neurons=512,256 => loss: 0.3551 - accuracy: 0.8735
                                                    # epochs=15 neurons=512.256 => loss: 0.3270 - accuracy: 0.8813
                                                    # epochs=30 neurons=512.256 => loss: 0.3020 - accuracy: 0.8932

CV_model3.evaluate(X_test, Y_test)              # epochs=10, neurons=512 => loss: 0.5512 - accuracy: 0.8224
                                                # epochs=10 neurons=512,256 => loss: 0.4706 - accuracy: 0.8498
                                                # epochs=15 neurons=512.256 => loss: 0.4383 - accuracy: 0.8612
                                                # epochs=30 neurons=512.256 => loss: 0.5021 - accuracy: 0.8659
                                                # NOTE: 30 epochs is overfitting, 15 is better with 512, 256 neurons
classifications = CV_model3.predict(X_test)
print(classifications[0])
print(Y_test[0])

# %%
# Better way of tuning parameters with callback function
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.1):
      print("\nReached desired loss so cancelling training!")
      self.model.stop_training = True

# %%
# Final Model without any convolution or pooling
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

CV_model4 = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])

CV_model4.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', metrics=["accuracy"])

CV_model4.fit(X_train, Y_train, epochs=30, callbacks=[callbacks])
# Result: execution was stopped at end of epoch 4/15 => loss < 0.3
CV_model4.evaluate(X_test, Y_test)

# %%
# Model with two layers of convolution

# first convo expects a tensor with all info => shape = (numImgs, x_shape, y_shape, z_shape)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# each convo layer has 64 filters each of size 3x3
CV_model5 = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPool2D(2, 2),   # maxPool with 2x2 scope
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])

CV_model5.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=["accuracy"])

CV_model5.summary()
callbacks = myCallback()

CV_model5.fit(X_train, Y_train, epochs=15, callbacks=[callbacks])
# epochs 5 => loss: 0.1702 - accuracy: 0.9352
# epochs 15 => loss: 0.0470 - accuracy: 0.9821
# Result: with callbacks execution was stopped at end of epoch 15/15 => loss < 0.05
CV_model5.evaluate(X_test, Y_test)          # loss: 0.2550 - accuracy: 0.9120

# Tested using only one convo layer, but turns out 2 does noticeably better
# Tested with 32 filters and it is not as powerful as 64 filters, but noticeably longer
