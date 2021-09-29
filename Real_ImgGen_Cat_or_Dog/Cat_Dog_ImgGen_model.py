import tensorflow as tf
import numpy as np

for i in [tf, np]:
    print(i.__name__, ": ", i.__version__, sep="")

# %%
import os

# To use unverified ssl you can add this to your code:
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Import data from online
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

dataset_path = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=dataset_url, extract=True)
dataset_dir = os.path.join(os.path.dirname(dataset_path), "cats_and_dogs_filtered")

print(dataset_path)
print(dataset_dir)

# Same as running the following in terminal
# wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#   -O /tmp/cats_and_dogs_filtered.zip

# %%
# Create subdirectories
train_cats_dir = os.path.join(dataset_dir, "train", "cats")
train_dogs_dir = os.path.join(dataset_dir, "train", "dogs")
validation_cats_dir = os.path.join(dataset_dir, "validation", "cats")
validation_dogs_dir = os.path.join(dataset_dir, "validation", "dogs")

train_dir = os.path.join(dataset_dir, "train")
validation_dir = os.path.join(dataset_dir, "validation")

# Show directories created
all_dirs = [train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir]
[print(i) for i in all_dirs]

# %%
# Show size of training and validation set
train_cats_num = len(os.listdir(train_cats_dir))
train_dogs_num = len(os.listdir(train_dogs_dir))

validation_cats_num = len(os.listdir(validation_cats_dir))
validation_dogs_num = len(os.listdir(validation_dogs_dir))

total_train = train_cats_num + train_dogs_num
total_validation = validation_dogs_num + validation_cats_num

print("train cats: ", train_cats_num)
print("train dogs: ", train_dogs_num)
print("validation cats: ", validation_cats_num)
print("validation_dogs: ", validation_dogs_num)
print("all train images: ", total_train)
print("all validation images: ", total_validation)

# %%
# Show what files names look like
train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

# %%
# Show Images
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0  # Index for iterating over images

# Set up matplotlib fig w/ size fit for 4x4 pics
fig = plt.gcf()  # get current figure
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index - 8: pic_index]
                ]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index - 8: pic_index]
                ]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)  # mpimg := math plot show img  in notebook
    plt.imshow(img)

plt.show()

# %%
# Data Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGen objects and rescale values so they are normalized (btwn 0 and 1)
train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Automatically label train and test data using flow_from_directory() 
train_generator = train_datagen.flow_from_directory(
    directory=train_dir, batch_size=20,
    class_mode='binary', target_size=(150, 150)
)
# Output: Found 2000 images belonging to 2 classes.

validation_generator = validation_datagen.flow_from_directory(
    directory=validation_dir, batch_size=20,
    class_mode='binary', target_size=(150, 150)
)
# Output: Found 1000 images belonging to 2 classes.

# %%
# Build model and configure (compile) training specifications

model = tf.keras.models.Sequential([
    # Note the desired input_shape for our imgs will be 150x150 with 3 channels for color
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),

    # Flatten and feed into deep NN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.RMSprop(lr=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

model.summary()

# %%
from time import time
# Fit the model => Perform training
# fit_generator is deprecated use fit()
t0 = time()
history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,   # 2,000 / 20 = 100
                              epochs=15,
                              validation_steps=50, # 1,000 / 20 = 50
                              verbose=2)
# steps_per_epoch = len(X_set) / batch_size
# steps_per_epoch := batch_num
# X_set:= validation or training

print((time()-t0) / 60, ' mins' ) # 397s ~6mins
# Output: epoch 15, 3x cov2D, 1 hidden => loss: 0.0298 - acc: 0.9915 - val_loss: 1.7559 - val_acc: 0.7090

# %%
# Import my own images to predict of dog or cat
from keras.preprocessing import image


test_image = image.load_img('luna.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
# %%
import matplotlib.pyplot as plt

# Retrieve a list of results on training and test data sets for each training epoch
accuracy = history.history['acc']
val_accuracy = history.history[ 'val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))  # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot(epochs, accuracy)
plt.plot(epochs, val_accuracy)
plt.title('Training vs Validation Accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training vs Validation Loss')

plt.show()
# Here we see that after 6 epochs the loss of the validation paltues
# while the loss of the training nears 0 => accuracy 100%
# Hence, we are overfitting as a result of a small restricted dataset
# Apply Image Agumentation to improve fit

# %%
# Add Image Agumention and rerun model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

dataset_path1 = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=dataset_url, extract=True)
dataset_dir1 = os.path.join(os.path.dirname(dataset_path1), "cats_and_dogs_filtered")

train_cats_dir1 = os.path.join(dataset_dir1, "train", "cats")
train_dogs_dir1 = os.path.join(dataset_dir1, "train", "dogs")
validation_cats_dir1 = os.path.join(dataset_dir1, "validation", "cats")
validation_dogs_dir1 = os.path.join(dataset_dir1, "validation", "dogs")

train_dir1 = os.path.join(dataset_dir1, "train")
validation_dir1 = os.path.join(dataset_dir1, "validation")

train_datagen1 = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen1 = ImageDataGenerator(rescale=1./255)

train_generator1 = train_datagen1.flow_from_directory(
        train_dir1,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator1 = test_datagen1.flow_from_directory(
        validation_dir1,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

t0 = time()

#model.fit_generator() depreciated => model.fit()
history1 = model.fit_generator(
      train_generator1,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=20,
      validation_data=validation_generator1,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)

print((time()-t0) / 60, ' mins' ) # 49.1415 mins for 100 epochs
# %%
import matplotlib.pyplot as plt

# Show accuracy vs loss
acc = history1.history['acc']
val_acc = history1.history['val_acc']
loss = history1.history['loss']
val_loss = history1.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# The graphs show that we are no longer overfitting since as the accuracy
# of the training set increases so does the acc of the test set
# However, we see dramatic spikes in the data showing that we dont need
# 100 epochs to get best lost/acc of the test set.

# %%
# Try Transfer Learning for better performance
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model

# Terminal command to download weights from online
# wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Create an instance of the Inception model and then load pretrained weights
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all layers untrainable in pretrained model
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape) # shape: (None, 7, 7, 768)
last_output = last_layer.output

#%%
from tensorflow.keras.optimizers import RMSprop
# Build full model in addition to pretrained model

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense (1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])
# After building model run data import and data preprocessing code

#%%
# Train model
history = model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 20,
            validation_steps = 50,
            verbose = 2)

#%%
# Display results
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()