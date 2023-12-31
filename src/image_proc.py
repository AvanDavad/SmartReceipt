import cv2


img = cv2.imread("images/receipts/IMG_5447.JPG")
# img = cv2.imread("images/receipts/IMG_5424.JPG")
# img = cv2.imread("images/sample1.png")
# img = cv2.imread("images/sample2.png")

# template
template1 = cv2.imread("images/templates/T3.png")
h, w = template1.shape[:2]

# template matching
res = cv2.matchTemplate(img, template1, cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

bottom_right = (max_loc[0] + w, max_loc[1] + h)
img = cv2.rectangle(img, max_loc, bottom_right, 255, 2)
# img = cv2.circle(img, (max_loc[0] + 80, max_loc[1] + 80), radius=10, color=(255,0,0), thickness=2)
cv2.imwrite("test.jpg", img)


