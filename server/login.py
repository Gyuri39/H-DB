import yaml
import streamlit as st
from pathlib import Path
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import CredentialsError, ForgotError, Hasher, LoginError, RegisterError, ResetError, UpdateError
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore
import json
from io import StringIO
import tempfile

db_client = None
if not firebase_admin._apps:
	firebase_config = st.secrets["firebase"]
	cred_dict = {	
		"type": firebase_config["type"],
		"project_id": firebase_config["project_id"].replace('\\n','\n'),
		"private_key_id": firebase_config["private_key_id"],
		"private_key": firebase_config["private_key"],
		"client_email": firebase_config["client_email"],
		"client_id": firebase_config["client_id"],
		"auth_uri": firebase_config["auth_uri"],
		"token_uri": firebase_config["token_uri"],
		"auth_provider_x509_cert_url": firebase_config["auth_provider_x509_cert_url"],
		"client_x509_cert_url": firebase_config["client_x509_cert_url"],
		"universe_domain": firebase_config["universe_domain"]
	}
	buf = StringIO()
	json.dump(cred_dict, buf)
	buf.seek(0)
	json_data = json.load(buf)
	cred = credentials.Certificate(json_data)
	initialize_app(cred)

db_client = firestore.client()
config_doc = db_client.collection("config").document("authenticator").get().to_dict()
#yaml_config = yaml.dump(config)
yaml_config = config_doc.get("yaml")
if yaml_config is None:
	raise NotImplementedError("Cannot find 'yaml' data from the database")

yaml_string = yaml.dump(yaml_config)

with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as config_file:
	config_file.write(yaml_string.encode("utf-8"))
	config_filename = config_file.name

with open(config_filename, 'r', encoding='utf-8') as f:
	config_dict = yaml.safe_load(f)

def upload_config(config_file):
	with open(config_file.name, 'r', encoding='utf-8') as f:
		new_config_data = yaml.safe_load(f)
	db_client.collection("config").document("authenticator").set({"yaml":new_config_data})

#filepath = Path('server/') / 'config.yaml'
def OpenAuthenticator():
	authenticator = stauth.Authenticate(config_filename)
#	authenticator = stauth.Authenticate(
#		credentials = config["credentials"],
#		cookie = config["cookie"],
#		preauthroized = config["pre-authorized"]
#	)
	return authenticator

def LoginWidget(authenticator):
#	authenticator = OpenAuthenticator()
	try:
		authenticator.login(location = "main", max_login_attempts = 10, single_session = False)
	except LoginError as e:
		st.error(e)

def UserRegisterWidget(authenticator):
#	authenticator = OpenAuthenticator()
	try:
		(email_of_registered_user,
			username_of_registered_user,
			name_of_registered_user) = authenticator.register_user(pre_authorized=config_dict['pre-authorized'], captcha=True)
		if email_of_registered_user:
			st.success('User registered successfully')
			upload_config(config_file)
	except RegisterError as e:
		st.error(e)

def PasswordResetWidget(authenticator):
#	authenticator = OpenAuthenticator()
	if st.session_state['authentication_status']:
		try:
			if authenticator.reset_password(st.session_state['username']):
				st.success('Password modified successfully')
				upload_config(config_file)
		except Exception as e:
			st.error(e)
	else:
		raise NotImplementedError("Authentication should precede")

def ForgotPasswordWidget(authenticator):
#	authenticator = OpenAuthenticator()
	try:
		(username_of_forgotten_password,
			email_of_forgotten_password,
			new_random_password) = authenticator.forgot_password(captcha = True, send_email = True, two_factor_auth = True)
		if username_of_forgotten_password:
			st.success('New password sent securely')
			upload_config(config_file)
	#		st.warning(new_random_password)
		elif not username_of_forgotten_password:
			st.error('Username not found')
	except ForgotError as e:
		st.error(e)

def ForgotUsernameWidget(authenticator):
#	authenticator = OpenAuthenticator()
#	cookie_manager = stx.Cookiemanger(key="cookie_manager_forgot_username")
#	authenticator = OpenAuthenticator()
	try:
		(username_of_forgotten_username,
			email_of_forgotten_username) = authenticator.forgot_username(captcha = True)
		if username_of_forgotten_username:
			st.success(f'Your username is "{username_of_forgotten_username}".')
			upload_config(config_file)
		elif not username_of_forgotten_username:
			st.error('Email not found')
	except ForgotError as e:
		st.error(e)

def UpdateUserDetailsWidget(authenticator):
#	authenticator = OpenAuthenticator()
	if st.session_state['authentication_status']:
		try:
			if authenticator.update_user_details(st.session_state['username']):
				st.success('Entry updated successfully')
				upload_config(config_file)
		except UpdateError as e:
			st.error(e)

def LogoutWidget(authenticator):
	if st.session_state['authentication_status']:
#		authenticator = OpenAuthenticator()
		authenticator.logout()
	else:
		raise NotImplementedError("Authentication should precede")
		
