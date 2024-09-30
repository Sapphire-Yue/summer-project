import tensorflow as tf
import label_define as ld
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

train_generator = ld.train_binarized_generator
validation_generator = ld.validation_binarized_generator

print(ld.train_generator.class_indices)

# 定義類別名稱
class_names = ['Gesture down', 'Gesture left', 'Gesture right', 'Gesture up']  # Updated to 4 classes

# 建立模型
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (5, 5), activation='relu'),
    BatchNormalization(),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='softmax')  # Updated to 4 classes
])

# 編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 顯示模型架構
model.summary()

# 訓練模型
history = model.fit(
    train_generator,
    steps_per_epoch=ld.train_generator.samples // ld.train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=ld.validation_generator.samples // ld.validation_generator.batch_size,
    epochs=20
)

# 使用驗證集進行模型評估
validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=ld.validation_steps)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# 儲存模型
model.save('hand_gesture_model.h5')


# 預測部分
def plot_image(predicted_class, true_class, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    color = 'blue' if predicted_class == true_class else 'red'
    plt.xlabel(f"Predicted: {class_names[predicted_class]} (True: {class_names[true_class]})", color=color)

def plot_value_array(predictions, true_class):
    plt.grid(False)
    plt.xticks(range(4))  # Updated to 4 classes
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions, color="#777777")  # Updated to 4 classes
    plt.ylim([0, 1])
    predicted_class = np.argmax(predictions)

    thisplot[predicted_class].set_color('red')
    thisplot[true_class].set_color('blue')


# 預測測試集數據
test_images, test_labels = next(validation_generator)  # 取得一個批次的測試數據

for i in range(20):  # 預測 20 張圖片
    img = test_images[i]  # 從測試集中提取圖片
    true_label = np.argmax(test_labels[i])  # 取得真實標籤

    # 進行預測
    img_array = np.expand_dims(img, axis=0)  # 增加 batch 維度
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # 取得預測類別索引

    # 可視化
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(predicted_class, true_label, img)

    plt.subplot(1, 2, 2)
    plot_value_array(predictions[0], true_label)
    plt.show()

    print(f"Predicted class: {class_names[predicted_class]} (True: {class_names[true_label]})")
