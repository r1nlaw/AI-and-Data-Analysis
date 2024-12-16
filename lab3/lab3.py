import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
import seaborn as sn

# Загрузка и нормализация данных MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Нормализуем изображения
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# Отображаем первое нормализованное изображение
plt.imshow(x_train_normalized[0], cmap=plt.cm.binary)
plt.show()

# Выводим размеры данных
print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Отображаем сетку из 25 изображений
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i + 1)
    plt.xticks([]), plt.yticks([]), plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

# Строим модель MLP (многослойный перцептрон)
model = tf.keras.models.Sequential([
    # Входной слой
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Первый скрытый слой
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
    # Второй скрытый слой
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
    # Выходной слой
    tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.002))
])

# Отображаем структуру модели
model.summary()

# Компилируем модель
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Настроим callback для TensorBoard
log_dir = ".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Обучаем модель
history = model.fit(x_train_normalized, y_train, epochs=5, validation_data=(x_test_normalized, y_test), callbacks=[tensorboard_callback])

# Графики потерь и точности
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Потери
axs[0].plot(history.history['loss'], label='Training Loss')
axs[0].plot(history.history['val_loss'], label='Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Точность
axs[1].plot(history.history['accuracy'], label='Training Accuracy')
axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.show()

# Оцениваем модель
train_loss, train_accuracy = model.evaluate(x_train_normalized, y_train)
print(f'Training loss: {train_loss}')
print(f'Training accuracy: {train_accuracy}')

validation_loss, validation_accuracy = model.evaluate(x_test_normalized, y_test)
print(f'Validation loss: {validation_loss}')
print(f'Validation accuracy: {validation_accuracy}')

# Сохраняем модель
model.save('digits_recognition_mlp.h5')

# Загружаем модель и делаем предсказания
loaded_model = tf.keras.models.load_model('digits_recognition_mlp.h5')
predictions_one_hot = loaded_model.predict(x_test_normalized)

# Преобразуем предсказания в метки классов
predictions = np.argmax(predictions_one_hot, axis=1)

# Отображаем предсказания для нескольких изображений
plt.figure(figsize=(10, 10))
num_cells = math.ceil(math.sqrt(25))
for i in range(25):
    plt.subplot(num_cells, num_cells, i + 1)
    plt.xticks([]), plt.yticks([]), plt.grid(False)
    plt.imshow(x_test_normalized[i], cmap=plt.cm.binary)
    plt.xlabel(f'Pred: {predictions[i]}')
plt.show()

# Строим матрицу ошибок
conf_matrix = tf.math.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9, 7))
sn.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5, square=True)
plt.title('Confusion Matrix')
plt.show()

# Запуск TensorBoard (в терминале):
# tensorboard --logdir=./logs/fit
