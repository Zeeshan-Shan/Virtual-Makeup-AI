import cv2
cap = cv2.VideoCapture(0)
cv2.namedWindow("Picture Saver")
img_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k == ord("q"):
        # q pressed
        break
    elif k == ord("c"):
        # c pressed
        save_path = "C:\\Virtual-Makeup-main\\Images\\"
        img_name = save_path + f"image{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print("Success")
        img_counter += 1

cap.release()

cv2.destroyAllWindows()
