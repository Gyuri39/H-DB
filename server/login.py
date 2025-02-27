import yaml
import streamlit as st
from pathlib import Path
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import CredentialsError, ForgotError, Hasher, LoginError, RegisterError, ResetError, UpdateError

filepath = Path('server/') / 'config.yaml'
with open(filepath, 'r', encoding='utf-8') as file:
	config = yaml.load(file, Loader=SafeLoader)

def OpenAuthenticator():
	authenticator = stauth.Authenticate(str(filepath))
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
			name_of_registered_user) = authenticator.register_user(pre_authorized=config['pre-authorized']['emails'], captcha=True)
		if email_of_registered_user:
			st.success('User registered successfully')
	except RegisterError as e:
		st.error(e)

def PasswordResetWidget(authenticator):
#	authenticator = OpenAuthenticator()
	if st.session_state['authentication_status']:
		try:
			if authenticator.reset_password(st.session_state['username']):
				st.success('Password modified successfully')
		except Exception as e:
			st.error(e)
	else:
		raise NotImplementedError("Authentication should precede")

def ForgotPasswordWidget(authenticator):
#	authenticator = OpenAuthenticator()
	try:
		(username_of_forgotten_password,
			email_of_forgotten_password,
			new_random_password) = authenticator.forgot_password(captcha = True)
		if username_of_forgotten_password:
			st.success('New password sent securely')
			st.warning(new_random_password)
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
		except UpdateError as e:
			st.error(e)

def LogoutWidget(authenticator):
	if st.session_state['authentication_status']:
#		authenticator = OpenAuthenticator()
		authenticator.logout()
	else:
		raise NotImplementedError("Authentication should precede")
		
