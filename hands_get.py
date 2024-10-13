import cv2
import numpy as np

# 啟動攝像頭
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 翻轉影像，避免左右顛倒
    frame = cv2.flip(frame, 1)

    # 設定抓取手部的區域，例如右下角
    height, width, _ = frame.shape
    x_start = int(width * 0.75)  # 從影像寬度的 50% 處開始
    y_start = int(height * 0.75)  # 從影像高度的 50% 處開始
    region_of_interest = frame[y_start:, x_start:]

    # 將ROI轉換為HSV色彩空間
    hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)

    # 設定膚色範圍 (HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)    # 膚色的最小值 (根據需要調整)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)  # 膚色的最大值 (根據需要調整)

    # 根據設定的範圍產生膚色遮罩
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 使用形態學操作來去除雜訊
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 高斯模糊
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 找出邊框 (輪廓)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到輪廓，繪製手部的邊框
    if len(contours) > 0:
        # 找到面積最大的輪廓
        max_contour = max(contours, key=cv2.contourArea)

        # 畫出最外層的邊框
        x, y, w, h = cv2.boundingRect(max_contour)
        # 注意: 繪製的框架需要偏移到完整的影像座標系
        cv2.rectangle(frame, (x_start + x, y_start + y), (x_start + x + w, y_start + y + h), (0, 255, 0), 2)

        # 使用凸包進行邊界檢測
        hull = cv2.convexHull(max_contour)
        # 繪製凸包邊界，注意同樣需要偏移
        cv2.drawContours(frame[y_start:, x_start:], [hull], -1, (255, 0, 0), 2)

    # 在原影像上顯示ROI框
    cv2.rectangle(frame, (x_start, y_start), (width, height), (255, 255, 255), 2)
    
    # 顯示結果影像
    cv2.imshow("Hand Detection in ROI", frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉所有窗口
cap.release()
cv2.destroyAllWindows()
