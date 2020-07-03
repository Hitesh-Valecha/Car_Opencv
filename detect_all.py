import cv2

cascade_car = 'cascade/cars.xml'
cascade_bike = 'cascade/wheel2.xml'
cascade_person = 'cascade/person_walking.xml'
cascade_bus = 'cascade/bus.xml'

video_car = 'dataset/cars.mp4'
video_bike = 'dataset/wheel2.mp4'
video_person = 'dataset/person.avi'
video_bus = 'dataset/bus1.mp4'

# cap = cv2.VideoCapture(video_car)
# vehicle_cascade = cv2.CascadeClassifier(cascade_car)

# cap = cv2.VideoCapture(video_bus)
# vehicle_cascade = cv2.CascadeClassifier(cascade_bus)

cap = cv2.VideoCapture(video_bike)
vehicle_cascade = cv2.CascadeClassifier(cascade_bike)

# cap = cv2.VideoCapture(video_person)
# vehicle_cascade = cv2.CascadeClassifier(cascade_person)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = vehicle_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        roi_gray = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)   #ROI is region of interest
        img_item = "1.png"
        cv2.imwrite(img_item, roi_gray)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()