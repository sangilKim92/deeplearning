import cv2
import numpy as np

img = cv2.imread('./110-150.jpg')

imgResize = cv2.resize(img,(1000,500))
imgCropped = img[:200,100:300]


print(img.shape)
print(imgResize.shape)
print(imgCropped.shape)

img2 = np.zeros((512,512,3),np.uint8) #0~255
img2[:] = 255,0,0 # 이미지 전체에 파란색 부여
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
cv2.rectangle(img, (0,0), (250,100),(0,0,255),cv2.FILLED)
cv2.putText(img, "OPENCV", (100,0),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)


cv2.imshow('Image',img)


cv2.waitKey(0)