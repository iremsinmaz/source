import tensorflow as tf


(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

model = tf.keras.models.Sequential([tf.keras.Layers.Flatten(),
                                    tf.keras.Layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.Layers.Dense(10, activation=tf.nn.softmax)])
