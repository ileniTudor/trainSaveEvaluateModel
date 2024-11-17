import os

import cv2
import inference

ROBOFLOW_API_KEY="your key"
os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY
img = cv2.imread("180_2247.jpg")

model = inference.get_model("dentalxray-s3wqb-5bdef/1")
segmentation_res = model.infer(image="180_2247.jpg")

for pred in segmentation_res[0].predictions:
    print("confidence:", pred.confidence)
    print("class:", pred.class_id)
    print("class name:", pred.class_name)
    cv2.rectangle(img, (int(pred.x-pred.width/2), int(pred.y-pred.height/2)), (int(pred.x+pred.width/2), int(pred.y+pred.height/2)), (0, 255, 0), 2 )
    cv2.imshow("img",img)
    cv2.waitKey(0)
