import cv2
from roboflow_segmentation_locally import Roboflow
rf = Roboflow(api_key="WD1Mevk49Aalrm1m7oCK")
project = rf.workspace().project("sem4-q9sys")
model = project.version(1).model


img = cv2.imread("your_image.jpg")

# infer on a local image
res = model.predict("your_image.jpg", confidence=10, overlap=30).json()

for pred in res["predictions"]:
    x = int(pred['x'])
    y = int(pred['x'])
    width = int(pred['width'])
    height = int(pred['height'])
    cv2.rectangle(img, (x, y), (x + width, y+width), (255,0,0), 2)

cv2.imshow("res", img)
cv2.waitKey(0)
print(res)

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())