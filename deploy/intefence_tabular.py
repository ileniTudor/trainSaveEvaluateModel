import requests
url = "https://api.obviously.ai/v3/model/automl/predict/single/04ed4ad0-7d3b-11ee-95a6-a7231b034929"

payload = {
	"gender": "Female",
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
	"TotalCharges": "29.85"
}

headers = {
   "Accept": "application/json",
   "Authorization": "ApiKey 1a52266a-7d3b-11ee-9e0f-1a025ee0ca8d",
   "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.text)
print("done")