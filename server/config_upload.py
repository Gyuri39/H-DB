import firebase_admin
from firebase_admin import credentials, firestore, storage
import yaml

with open("config.yaml", 'r') as file:
	config = yaml.safe_load(file)

cred = credentials.Certificate("../.streamlit/firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
db.collection("config").document("authenticator").set({"yaml":config})
