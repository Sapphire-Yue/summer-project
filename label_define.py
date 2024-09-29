import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cvt_color import preprocess_and_binarize

def custom_binarized_generator(generator):
    """
    自定義生成器，用於對批次中的圖片進行二值化處理
    :param generator: 原始的圖像生成器
    :return: 二值化處理後的圖像批次和標籤
    """
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
print(train_generator.class_indices)
