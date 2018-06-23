import Augmentor
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import dataset
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.nan)

def categorical_labels(numerical_labels):
    """
    Return categorical labels for an array of 0-based numerical labels.
    :param numerical_labels: The numerical labels.
    :type numerical_labels: Array-like list.
    :return: The categorical labels.
    """
    # class_labels_np = np.array([x.class_label_int for x in numerical_labels])
    class_labels_np = np.array(numerical_labels)
    one_hot_encoding = np.zeros((class_labels_np.size, class_labels_np.max() + 1))
    one_hot_encoding[np.arange(class_labels_np.size), class_labels_np] = 1
    one_hot_encoding = one_hot_encoding.astype(np.uint)

    return one_hot_encoding


train_path = 'real/train/'
test_path = 'real/test/'

# train_path = 'ddata/train/'
# test_path = 'ddata/test/'
# checkpoint_dir = "ckpts/"

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 1

# image dimensions (only squares for now)
img_size = 4

# class info
classes = ['0', '1', '2']
num_classes = len(classes)

# validation split
validation_size = .2

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

x_train = data.train.images
y_train = data.train.labels.tolist()
x_test = test_images
y_test = data.valid.labels

y_train = categorical_labels(data.train.labels)
y_test = categorical_labels(data.valid.labels)

p = Augmentor.Pipeline()
# p.rotate(probability=1, max_left_rotation=3, max_right_rotation=3)

np.shape(x_train[0])
plt.imshow(x_train[0], cmap="Greys")
# plt.show()

num_classes = 3
input_shape = (4, 4, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.Adadelta(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

batch_size = 128
g = p.keras_generator_from_array(x_train, y_train, batch_size=4)
X, y = next(g)
plt.imshow(X[0].reshape(4,4), cmap="Greys")
# plt.show()
# print("The image above has the label %s."  % int(np.nonzero(y[0])[0]))

h = model.fit_generator(g, steps_per_epoch=len(x_train)/batch_size, epochs=180, verbose=1)