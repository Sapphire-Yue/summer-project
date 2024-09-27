import tensorflow as tf
import label_define as ld
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

train_generator = ld.train_generator
validation_generator = ld.validation_generator

print(train_generator.class_indices)

# 定義類別名稱
class_names = ['Gesture down', 'Gesture enter', 'Gesture left', 'Gesture right', 'Gesture stop', 'Gesture up']  # 根據你的實際手勢數據

# 建立模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout 防止過擬合
    Dense(6, activation='softmax')  # 最後輸出層，5 個手勢類別（根據具體數量調整）
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 顯示模型架構
model.summary()


# 訓練模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20)


# 使用驗證集進行模型評估
validation_loss, validation_accuracy = model.evaluate(validation_generator)
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
    plt.xticks(range(6))  # 假設6個類別
    plt.yticks([])
    thisplot = plt.bar(range(6), predictions, color="#777777")
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



# # 預測12次
# for i in range(12):
#     # 載入影像
#     img = image.load_img('hand_image'+ str(i) +'.jpg', target_size=(64, 64))
#     img_array = image.img_to_array(img) / 255.0  # 將影像轉為 numpy array 並標準化
#     img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 維度

#     # 進行預測
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions)  # 取得類別索引
#     print(f"Predicted class: {predicted_class}")
