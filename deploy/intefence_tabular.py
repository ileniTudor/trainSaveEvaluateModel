import requests
url = "https://api.obviously.ai/v3/model/automl/predict/single/05bc7020-8a22-11ee-b3a7-b7ae08b39965"

payload = {
	"SeniorCitizen": "0",
	"Partner": "Yes",
	"Dependents": "No",
	"tenure": "1.0",
	"PhoneService": "No",
	"MultipleLines": "No phone service",
	"InternetService": "DSL",
	"OnlineSecurity": "No",
	"OnlineBackup": "Yes",
	"DeviceProtection": "No",
	"TechSupport": "No",
	"StreamingTV": "No",
	"StreamingMovies": "No",
	"Contract": "Month-to-month",
	"PaperlessBilling": "Yes",
	"PaymentMethod": "Electronic check",
	"MonthlyCharges": "29.85",
	"TotalCharges": "29.85",
	"Churn": "No"
}

headers = {
   "Accept": "application/json",
   "Authorization": "ApiKey 7664f7ee-8a0f-11ee-9530-6eecdea147e5",
   "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.text)