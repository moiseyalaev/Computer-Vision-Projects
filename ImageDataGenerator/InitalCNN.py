import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# ============================================= Step 1: Data Preprocessing =============================================

# ============ Training Set ==============
# Image Augmentation on train set then
train_datagen = ImageDataGenerator(
    rescale=1/255, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64),
                                                    batch_size=32, class_mode='binary')

 # ============= Test Set =================
test_datagen = ImageDataGenerator(rescale=1/255) # only do feature on test set

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64),
                                                    batch_size=32, class_mode='binary')
# ================================================ Step 2: Building CNN ================================================

# ============ Initialising CNN ==============
cnn = tf.keras.models.Sequential()

# ============ Convolution ===============
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               input_shape=[64,64,3])) # only needed for first later to connect to input layer

# ============== Pooling ===============
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# ========= Another Convolution ===============
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# now pool again
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# =========== Flattening ===============
cnn.add(tf.keras.layers.Flatten())

# =========== Full Connection ===============
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# =========== Output Layer  ===============
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ================================================ Step 3: Training CNN ================================================
# =========== Compile CNN  ===============
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# =========== Train CNN on training set  ===============
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# ========================================== Step 4: Making Single Prediction ==========================================
test_img = image.load_img(
    path='dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64)
)
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0)

result = cnn.predict(test_img)
training_set.class_indices

prediction = 'dog' if result[0][0] == 1 else 'cat'

print(prediction)

