# Terminal Commands to download training and test set
# ( make sure in correct working directory when running)
wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
              --output-document=/tmp/rps.zip

wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
              -O /rps-test-set.zip

# %%
# Extract zipped files into tmp directory
# From which we will make subdirectories from
import os
import zipfile

local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

local_zip = '/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

# %%
# Make subdirectories for data

rock_dir = os.path.join('/tmp/rps/rock')
paper_dir = os.path.join('/tmp/rps/paper')
scissors_dir = os.path.join('/tmp/rps/scissors')

print("Number of training rock images: ", len(os.listdir(rock_dir)))
print("Number of training paper images: ", len(os.listdir(paper_dir)))
print("Number of training scissor images: ", len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_dir)

paper_files = os.listdir(paper_dir)
print(paper_files)

scissors_files = os.listdir(scissors_dir)
print(scissors_files)
# %%
# Visualize data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

index = 2

next_rock = [os.path.join(rock_dir, fname)
             for fname in rock_files[index-2: index]]

next_paper = [os.path.join(paper_dir, fname)
              for fname in paper_files[index-2: index]]

next_scissors = [os.path.join(scissors_dir, fname)
                 for fname in scissors_files[index-2: index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()

# %%
# Image Preprocessing using DataGen
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

training_dir = '/tmp/rps/'

training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_dir = "/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    training_dir,
    target_size=(150,150),
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    class_mode='categorical')

# %%
# Build Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


model = tf.keras.models.Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

# %%
# Fit model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    verbose=1)

model.save("rps.h5")

# %%
# Visualize results
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()