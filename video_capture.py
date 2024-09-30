import cv2
import numpy as np
import tensorflow.keras as keras

# 定義手勢類別名稱（根據你的模型訓練的手勢類別）
class_names = ['Gesture down', 'Gesture enter', 'Gesture left', 'Gesture right', 'Gesture stop', 'Gesture up']  # 替換成實際的手勢類別

# 載入模型
model = keras.models.load_model('hand_gesture_resnet_model.h5')

# 啟動攝像頭
cap = cv2.VideoCapture(0)

# 偵測的區域
x_start, y_start = 125, 125  # 偵測區域的左上角座標
width, height = 175, 175      # 偵測區域的寬度與高度

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 在畫面上繪製偵測區域的矩形框
    cv2.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), (0, 255, 0), 2)

    # 擷取偵測區域的影像部分
    cropped_frame = frame[y_start:y_start + height, x_start:x_start + width]
    # 預處理影像

    # 轉換為灰階影像
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # 雙邊濾波
    blurred_image = cv2.bilateralFilter(gray_frame, d=9, sigmaColor=75, sigmaSpace=75)

    # 邊緣檢測
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=25)

    # 增加線條粗細的膨脹操作
    kernel = np.ones((3, 3), np.uint8)  # 3x3 核，增加邊緣粗細
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # 膨脹一次，可以調整 iterations 增加粗細

    # 調整大小以符合模型輸入
    resized_edges = cv2.resize(dilated_edges, (64, 64))

    # 將灰階影像轉換為 3 通道
    edges_rgb = cv2.cvtColor(resized_edges, cv2.COLOR_GRAY2RGB)  # 轉換為 (64, 64, 3)

    # 增加 batch 維度並標準化
    img_array = np.expand_dims(edges_rgb, axis=0)  # (1, 64, 64, 3)
    img_array = img_array / 255.0  # 標準化像素值到 [0, 1]

    # 預測手勢
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # 取得預測類別索引
    predicted_label = class_names[predicted_class]  # 對應類別名稱

    # 在畫面上顯示預測結果
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 顯示每個手勢類別的機率
    for i, class_name in enumerate(class_names):
        probability = predictions[0][i]  # 取得每個類別的預測機率
        text = f"{class_name}: {probability:.2f}"
        cv2.putText(frame, text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 顯示邊緣檢測的結果
    cv2.imshow('Edge Detection', edges)
    
    # 顯示攝影機畫面
    cv2.imshow('Hand Gesture Recognition', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) == ord('q'):
        break

# 釋放攝像頭資源
cap.release()
cv2.destroyAllWindows()
