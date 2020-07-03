import cv2

cascade_src = 'cascade/cars.xml'
video_src = 'dataset/cars.mp4'
#video_src = 'dataset/video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        roi_gray = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)   #ROI is region of interest
        img_item = "1.png"
        cv2.imwrite(img_item, roi_gray)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()