from tensorflow import keras
import numpy as np
from PIL import Image

# load keras' images dataset for handwritten recognition
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# normalization for better performance while training
X_train = X_train / 255
X_test = X_test / 255

# create a neural network to handle the predictions
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"]
              )

model.fit(X_train, y_train, epochs = 10)
model.evaluate(X_test, y_test)

img = Image.open('images/number-5.jpg')
img = img.convert('L')
img = img.resize((28, 28))
img_array = np.array(img)
img_array = img_array.reshape(1, 28, 28)  # add batch dimension
img_array = img_array / 255  # normalize like training data


y_predicted = model.predict(img_array)
predicted_num = np.argmax(y_predicted[0])
print(predicted_num)