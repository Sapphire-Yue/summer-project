import cv2
import numpy as np

# 進行圖片預處理和二值化，並抓取手部輪廓
def preprocess_and_binarize(image):
    if image is None:
        print("Error: Could not load image.")
        return None
    
    # 將 float32 圖像轉換為 uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # 將圖像轉換為 HSV 色彩空間
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定義膚色的 HSV 範圍
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)    # 膚色範圍的下限
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)  # 膚色範圍的上限

    # 根據膚色範圍生成遮罩
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # 使用形態學操作去除小噪聲
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 找到邊緣
    edges = cv2.Canny(mask, threshold1=50, threshold2=25)

    # 找出輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 畫出輪廓
    contour_image = image.copy()
    if contours:
        # 將最大的輪廓抓取出來，並繪製凸包
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)
        cv2.drawContours(contour_image, [hull], -1, (0, 255, 0), 2)  # 綠色邊框
    
    return contour_image


# 測試：加載圖片並進行預處理和輪廓檢測
img = cv2.imread('dataset/down/Rdown_291.jpg')  # 替換為你的圖片路徑
result_img = preprocess_and_binarize(img)

# 顯示原圖與處理後的結果
cv2.imshow('Original Image', img)
cv2.imshow('Hand Contour Image', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
