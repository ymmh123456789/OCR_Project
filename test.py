import cv2

if __name__ == "__main__":
    img = cv2.imread("1111.png", 0)
    ret, pic = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("tmp", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()