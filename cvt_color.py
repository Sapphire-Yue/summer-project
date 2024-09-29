import cv2
import numpy as np

# 進行圖片二值化
def preprocess_and_binarize(image):
    if image is None:
        print("Error: Could not load image.")
        return None
    
    # 將 float32 圖像轉換為 uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # 轉換為灰度圖像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊去除雜訊
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Canny 邊緣檢測
    edges = cv2.Canny(blurred_image, threshold1=150, threshold2=100)
    
    # 將邊緣圖像轉換為三通道圖像以便於輸出
    final_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0  # 將數據標準化到 [0, 1]
    
    return final_image

    
"""
# 測試：加載圖片並進行預處理
img = cv2.imread('dataset/down/Rdown_291.jpg')
binarized_img = preprocess_and_binarize(img)

# 顯示原圖與處理後的二值化圖像
cv2.imshow('Original Image', img)
cv2.imshow('Binarized Image', binarized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""img = cv2.imread('dataset/down/Rdown_292.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); # 轉換前，都先將圖片轉換成灰階色彩
output = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('oxxostudio', img)
cv2.imshow('oxxostudio2', output)
cv2.waitKey(0)
cv2.destroyAllWindows()"""