from roboflow import Roboflow
rf = Roboflow(api_key="OV5Lde28GJ9sksEJLT1g")

project = rf.workspace().project("dentalxray-s3wqb-5bdef")
model = project.version(1).model

# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

model.predict("180_2247.jpg").save("roboflow_prediction.jpg")
