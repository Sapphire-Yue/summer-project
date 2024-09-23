from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# 檢查資料夾結構中的標籤
print(train_generator.class_indices)
