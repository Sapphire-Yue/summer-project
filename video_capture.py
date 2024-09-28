import cv2
import numpy as np
import tensorflow.keras as keras

# 定義手勢類別名稱（根據你的模型訓練的手勢類別）
class_names = ['Gesture down', 'Gesture enter', 'Gesture left', 'Gesture right', 'Gesture stop', 'Gesture up']  # 替換成實際的手勢類別

# 載入模型
model = keras.models.load_model('hand_gesture_model.h5')

# 啟動攝像頭
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 預處理影像
    img = cv2.resize(frame, (64, 64))  # 調整大小以符合模型輸入
    img_array = np.expand_dims(img, axis=0)  # 增加 batch 維度
    img_array = img_array / 255.0  # 標準化像素值到 [0, 1]

    # 預測手勢
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # 取得預測類別索引
    predicted_label = class_names[predicted_class]  # 對應類別名稱

    # 在畫面上顯示預測結果
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 顯示攝影機畫面
    cv2.imshow('Hand Gesture Recognition', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) == ord('q'):
        break

# 釋放攝像頭資源
cap.release()
cv2.destroyAllWindows()
