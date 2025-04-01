import cv2
import dlib
import numpy as np

def empty(a):
    pass

cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 400, 240)
cv2.createTrackbar("Blue", "BGR", 0, 255, empty)
cv2.createTrackbar("Green", "BGR", 0, 255, empty)
cv2.createTrackbar("Red", "BGR", 0, 255, empty)
cv2.createTrackbar("Brightness", "BGR", 10, 50, empty)
cv2.createTrackbar("Contrast", "BGR", 10, 30, empty)
cv2.createTrackbar("Denoise", "BGR", 0, 10, empty)

def create(img, points, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
    if cropped:
        x, y, w, h = cv2.boundingRect(points)
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, 5, 5)
        return imgCrop
    else:
        return mask

def brighten_skin(img, alpha, beta):
    img_bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img_bright

def denoise_skin(img, strength):
    return cv2.fastNlMeansDenoisingColored(img, None, strength * 5, strength * 5, 7, 21)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    img = cv2.imread("C:\Virtual-Makeup-main\Images\image2.jpg")
    if img is None:
        print("Error: Image not found! Check the path.")
        break
    img = cv2.resize(img, (0, 0), None, 2, 2)
    imgOriginal = img.copy()
    
    alpha = cv2.getTrackbarPos("Contrast", "BGR") / 10.0
    beta = cv2.getTrackbarPos("Brightness", "BGR")
    denoise_strength = cv2.getTrackbarPos("Denoise", "BGR")
    
    imgBrightened = brighten_skin(imgOriginal, alpha, beta)  # Apply skin brightening
    imgDenoised = denoise_skin(imgBrightened, denoise_strength)  # Apply noise reduction
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    
    for face in faces:
        landmarks = predictor(imgGray, face)
        mypoints = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])
        
        lips = create(img, mypoints[48:61], masked=True, cropped=False)
        
        imgColor = np.zeros_like(lips)
        b = cv2.getTrackbarPos("Blue", "BGR")
        g = cv2.getTrackbarPos("Green", "BGR")
        r = cv2.getTrackbarPos("Red", "BGR")
        imgColor[:] = b, g, r
        
        imgColor = cv2.bitwise_and(lips, imgColor)
        imgColor = cv2.GaussianBlur(imgColor, (9, 9), 20)
        
        imgDenoised = cv2.addWeighted(imgDenoised, 1, imgColor, 0.6, 0)  # Blend lips color with denoised image
    
    cv2.imshow("BGR", imgDenoised)
    cv2.imshow("Original", imgOriginal)
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()