import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import label_define as ld
import matplotlib.pyplot as plt
import numpy as np

# 使用定義的資料生成器
train_generator = ld.train_binarized_generator
validation_generator = ld.validation_binarized_generator

# 定義類別名稱
class_names = ['Gesture down', 'Gesture left', 'Gesture right', 'Gesture up']  # 根據你的實際手勢數據
#class_names = ['Gesture down', 'Gesture enter', 'Gesture left', 'Gesture right', 'Gesture stop', 'Gesture up']  # 根據你的實際手勢數據
num_classes = len(class_names)

# 載入 ResNet50 作為基礎模型，使用預訓練權重並排除頂層分類器
base_model = tf.keras.applications.ResNet50(
    include_top=False,  # 不包含原始的全連接層
    weights='imagenet',  # 使用 ImageNet 預訓練權重
    input_shape=(64, 64, 3)  # 定義輸入形狀
)

# 將 ResNet50 設置為不可訓練（鎖定預訓練權重）
base_model.trainable = False

# 定義新的模型架構
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # 全局平均池化，壓縮空間維度
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # 全連接層
    layers.Dropout(0.3),  # Dropout 防止過擬合
    layers.Dense(num_classes, activation='softmax')  # 對應手勢分類數量的輸出層
])

# 編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 顯示模型架構
model.summary()
"""
# 定義一個自訂的回調函數來顯示圖片
class ImageDisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 獲取一些驗證數據
        images, labels = next(iter(validation_generator))  # 取得第一個批次的驗證資料
        predictions = self.model.predict(images)  # 預測結果
        
        # 繪製圖片及其預測
        plt.figure(figsize=(10, 10))
        for i in range(9):  # 顯示前 9 張圖片
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])  # 顯示圖片
            plt.title(f"Predicted: {class_names[np.argmax(predictions[i])]}, True: {class_names[np.argmax(labels[i])]}")  # 顯示預測和真實標籤
            plt.axis('off')
        plt.show()"""

# 訓練模型
history = model.fit(
    train_generator,
    steps_per_epoch=ld.steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=ld.validation_steps,
    epochs=20,
    #callbacks=[ImageDisplayCallback()]  # 加入自訂回調函數
)

"""
# 假設已經完成第一輪訓練，我們來解凍更多的層
for layer in base_model.layers[-20:]:  # 解鎖 ResNet50 最後 40 層
    layer.trainable = True

# 再次編譯模型，使用較小的學習率來進行微調
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 使用更小的學習率
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 再次訓練
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=15  # 微調階段的訓練可以較少的 epoch 數
)
"""


# 使用驗證集進行模型評估
validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=ld.validation_steps)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# 儲存模型
model.save('hand_gesture_resnet_model.h5')
