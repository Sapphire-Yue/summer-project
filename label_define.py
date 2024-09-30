from cvt_color import preprocess_and_binarize
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

def custom_binarized_generator(generator):
    """
    Custom generator to apply binarization preprocessing to each batch of images.
    :param generator: Original image generator
    :return: Binarized images and labels
    """
    while True:
        # Get a batch of images and labels
        batch_x, batch_y = generator.next()
        # Apply binarization preprocessing to each image in the batch
        binarized_batch = np.array([preprocess_and_binarize(img) for img in batch_x])
        yield binarized_batch, batch_y

# Create an image data generator
datagen = ImageDataGenerator(
    rescale=1./255,          # 正規化
    #preprocessing_function=preprocess_and_binarize,  # Apply binarization preprocessing
    width_shift_range=0.2,   # 隨機水平位移
    height_shift_range=0.2,  # 隨機垂直位移
    shear_range=0.2,         # 隨機剪切變換
    zoom_range=0.2,          # 隨機縮放
    horizontal_flip=True,    # 隨機水平翻轉
    fill_mode='nearest',      # 空白區填補模式
    validation_split=0.2  # Split into training/validation sets
)

# 載入訓練數據
train_generator = datagen.flow_from_directory(
    'only_dataset/',  # 圖片資料夾
    target_size=(128, 128),  # 圖片縮放至 64x64
    batch_size=32,  # 批量大小
    class_mode='categorical',  # 多類別分類
    subset='training'  # 使用訓練集
)

# 載入驗證數據
validation_generator = datagen.flow_from_directory(
    'only_dataset/',  # 圖片資料夾
    target_size=(128, 128),  # 圖片縮放至 64x64
    batch_size=32,  # 批量大小
    class_mode='categorical',  # 多類別分類
    subset='validation'  # 使用驗證集
)

steps_per_epoch=train_generator.samples // train_generator.batch_size
validation_steps=validation_generator.samples // validation_generator.batch_size

# Use custom binarized generator
train_binarized_generator = custom_binarized_generator(train_generator)
validation_binarized_generator = custom_binarized_generator(validation_generator)
# 查看資料集中的分類標籤
print(train_generator.class_indices)

import matplotlib.pyplot as plt

# 提取一批訓練數據
images, labels = next(train_binarized_generator)

# 顯示前 9 張圖片
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i])  # 顯示圖片
    plt.title(f"Class: {np.argmax(labels[i])}")  # 顯示對應的分類標籤
    plt.axis('off')
plt.show()


"""import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cvt_color import preprocess_and_binarize

def custom_binarized_generator(generator):

    自定義生成器，用於對批次中的圖片進行二值化處理
    :param generator: 原始的圖像生成器
    :return: 二值化處理後的圖像批次和標籤

    while True:
        # 每次從生成器中獲取一批次的圖像和標籤
        batch_x, batch_y = generator.next()
        # 對每個批次的圖片進行二值化處理
        binarized_batch = np.array([preprocess_and_binarize(img) for img in batch_x])
        yield binarized_batch, batch_y


# 創建一個資料生成器
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 進行圖片縮放和切分訓練/驗證集

# 加載訓練資料
train_generator = datagen.flow_from_directory(
    'dataset/',  # 資料夾路徑
    target_size=(64, 64),  # 調整圖片大小為 64x64
    batch_size=32,  # 批次大小
    class_mode='categorical',  # 多類別分類
    subset='training'  # 使用訓練集
)

# 加載驗證資料
validation_generator = datagen.flow_from_directory(
    'dataset/',  # 資料夾路徑
    target_size=(64, 64),  # 調整圖片大小為 64x64
    batch_size=32,  # 批次大小
    class_mode='categorical',  # 多類別分類
    subset='validation'  # 使用驗證集
)

# 使用自定義生成器
train_binarized_generator = custom_binarized_generator(train_generator)
validation_binarized_generator = custom_binarized_generator(validation_generator)

# 檢查資料夾結構中的標籤
print(train_generator.class_indices)"""
