import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image

import os

x_array = []
y_array = []
dirs = {
    0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ё', 7: 'Ж', 8: 'З', 9: 'И', 10: 'Й', 11: 'К', 12: 'Л', 13: 'М',
    14: 'Н', 15: 'О', 16: 'П', 17: 'Р', 18: 'С', 19: 'Т', 20: 'У', 21: 'Ф', 22: 'Х', 23: 'Ц', 24: 'Ч', 25: 'Ш', 26: 'Щ',
    27: 'Ъ', 28: 'Ы', 29: 'Ь', 30: 'Э', 31: 'Ю', 32: 'Я'
}

for files in dirs:
    my_path = os.path.join('dataset', dirs[files])
    for img in os.listdir(my_path):
        image = Image.open(os.path.join(my_path, img)).convert('L')
        image = image.resize((28, 28), Image.ANTIALIAS)
        WIDTH, HEIGHT = image.size
        data = np.array(image.getdata())
        data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
        x_array.append(data)
        y_array.append(files)

x_array = np.asarray(x_array)
y_array = np.asarray(y_array)
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.25, random_state=42)
#
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 33)
y_test_cat = keras.utils.to_categorical(y_test, 33)


# отображение первых 25 изображений из обучающей выборки
# plt.figure(figsize=(10, 5))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#
# plt.show()

model = keras.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(33, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль
my_optimizer = keras.optimizers.Adam(learning_rate=0.001)
# my_optimizer = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=True)
model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=40, epochs=8, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

#
# n = 1
# x = np.expand_dims(x_test[n], axis=0)
# res = model.predict(x)
# print( res )
# print( np.argmax(res) )
#
# plt.imshow(x_test[n], cmap=plt.cm.binary)
# plt.show()
#
# # Распознавание всей тестовой выборки
# pred = model.predict(x_test)
# pred = np.argmax(pred, axis=1)
#
# print(pred.shape)
#
# print(pred[:20])
# print(y_test[:20])
#
# # Выделение неверных вариантов
# mask = pred == y_test
# print(mask[:10])
#
# x_false = x_test[~mask]
# y_false = x_test[~mask]
#
# print(x_false.shape)
#
# # Вывод первых 25 неверных результатов
# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_false[i], cmap=plt.cm.binary)
#
# plt.show()