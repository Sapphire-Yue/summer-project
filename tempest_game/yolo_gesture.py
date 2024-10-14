import time
import torch
import cv2
from PIL import Image
import config
import numpy as np
import pandas as pd

class YoloGesture:

    def __init__(self):
        # 載入 YOLOv5 模型，假設你已經有一個訓練好的手勢辨識模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/Python/pygame-summer-team-jam-main/best.pt')
        self.enabled = config._default_configs["Gesture"]["enabled"]
        # 檢查手勢映射
        """self.mapping = config._default_configs["Gesture"]["mapping"]
        print("Gesture mapping:", self.mapping)  # 確認手勢映射是否正確"""
        self.mapping = {
            "up": "jump",
            "left": "left",
            "right'": "right",
            "down": "slide",
            "enter": "enter",
            "stop": "stop"
        }
        # 開啟攝像頭
        self.cap = cv2.VideoCapture(0)

        # 設置計時器
        self.last_detection_time = time.time()  # 用來記錄上一次偵測的時間
        self.detection_interval = 0.2  # 設定每 0.3 秒回傳一次動作

    def detect_gesture(self):
        if not self.enabled:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None


        # 將攝像頭影像轉換為 PIL Image 格式
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 使用 YOLOv5 模型進行偵測
        results = self.model(img)
        # 將檢測結果渲染到原始影像上
        output = results.render()  # 注意這裡的 output 是 RGB 格式的列表

        # 將第一個渲染結果轉換為 numpy array 格式
        output = np.squeeze(output[0])  # 確保只取第一個檢測結果
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # 轉換顏色空間回 BGR 格式

        # 顯示即時辨識結果 (OpenCV)
        cv2.imshow('YOLOv5 Real-time Detection', output)

        # 假設 YOLO 模型回傳的是每個手勢的名稱和置信度
        detected_data = results.pandas().xyxy[0]
        #print(f"Detected Data: {detected_data}")  # 輸出檢測數據

        # 只選擇置信度大於 0.75 的檢測結果
        detected_gestures = []
        actions = []
        for index, row in detected_data.iterrows():
            if row['confidence'] > 0.75:  # 只選擇置信度大於 0.75 的結果
                gesture = row['name']
                #print(f"Detected gesture: {gesture}")
                detected_gestures.append(gesture)
                """if gesture in self.mapping:
                    print(f"Detected gesture: {gesture}")
                    actions.append(self.mapping[gesture])
                """
                if gesture in self.mapping:
                    action = self.mapping[gesture]
                    actions.append(action)
                    #print(f"Mapped action: {action}")
                else:
                    print(f"Gesture '{gesture}' not in mapping.")
                

        # 檢查時間間隔，是否已經過了 0.3 秒
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return None  # 若間隔不足 0.3 秒，則不進行偵測
        
        # 更新上一次偵測的時間
        self.last_detection_time = current_time
        
        return actions

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()  # 確保關閉所有 OpenCV 窗口
