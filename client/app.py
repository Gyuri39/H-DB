#Python path designation
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#Import streamlit
import streamlit as st
st.set_page_config(
	page_title = "Web UI",
	page_icon = "yin-yang",
	layout = "wide"
)
#Import packages
import datetime
from streamlit_option_menu import option_menu
from my_pages import ViewData, AddData, TrainModel, TestModel
from server.login import OpenAuthenticator, LoginWidget, LogoutWidget, UserRegisterWidget, ForgotUsernameWidget, ForgotPasswordWidget, PasswordResetWidget
import firebase_admin	# Firestore

def show_login_page():
	with st.sidebar:
		main_menu = option_menu("User Authentication", ["Login", "Register"],
			icons=["person-circle", "person-plus"], menu_icon="key", default_index=0)
		main_menu
	if main_menu == "Login":
		sub_menu = option_menu(
			menu_title="Login",
			options=["Sign in", "Find ID/PW"],
			menu_icon="person-circle",
			default_index=0,
			orientation="horizontal"
		)
		if sub_menu == "Sign in":
			st.title("Login to the system")
			LoginWidget(st.session_state.authenticator)
			if st.session_state.authentication_status:
				st.rerun()
			elif st.session_state.authentication_status == None:
				st.warning("Please log in to access the app.")
			elif st.session_state.authentication_status == False:
				st.error("Wrong ID/PW")
			else:
				raise ValueError("The authentication_status is not boolean.")
		elif sub_menu == "Find ID/PW":
			forgotten_info = st.selectbox("I want to find my ", ["Username", "Password"], index=None)
			if forgotten_info == "Username":
				ForgotUsernameWidget(st.session_state.authenticator)
			elif forgotten_info == "Password":
				ForgotPasswordWidget(st.session_state.authenticator)
	elif main_menu == "Register":
		st.title("Register to the system")
		st.write("Your e-mail must be pre-registered.\nContact us for pre-registeration.")
		UserRegisterWidget(st.session_state.authenticator)


def show_main_app():
	with st.sidebar:
		st.write(f"Logged in as: **{st.session_state.name}**")
		main_menu = option_menu("Main Menu", ["Home", "Data Management", "Modeling", "Option"],
			icons=['house', 'file-earmark', "graph-up", "gear"], menu_icon="cast", default_index=0)
		main_menu
		LogoutWidget(st.session_state.authenticator)

	if main_menu == "Home":
		now = datetime.datetime.now()
		st.markdown(
		f"""
		<div style="text-align: center; font-size: 20px;">
		<p style="font-size:50px;"><b>Hydrogen Dataset</b></p>
		<p></p>
		<p style="font-size:15px:">Version 0.3.4.0-beta</p>
		<p>Welcome to the beta test of our application!</p>
		<p>This version is under active development, and your feedback is invaluable to us.</p>
		<p>Please report any bugs or issues you encounter during testing.</p>
		<p></p>
		<p><b>Select a menu from the sidebar </b> to access the tools.</p>
		<p></p>
		</div>
		""",
		unsafe_allow_html=True
		)
		#if st.checkbox("Change Password"):
		#	PasswordResetWidget(st.session_state.authenticator)
		#else:
		#	st.empty()



	elif main_menu == "Data Management":
		sub_menu = option_menu(
			menu_title="Data Management",
			options=["View Data", "Add Data"],
			icons=["file-earmark-bar-graph", "file-earmark-arrow-up"],
			menu_icon="file-earmark",
			default_index=0,
			orientation="horizontal"
		)
		if sub_menu == "View Data":
			ViewData.createPage()
		elif sub_menu == "Add Data":
			AddData.createPage()

	elif main_menu == "Modeling":
		sub_menu = option_menu(menu_title = "Modeling",
			options=["Generate Model", "Apply Model"],
			icons=["sliders", "android2"],
			menu_icon="graph-up",
			default_index=0,
			orientation="horizontal"
		)
		if sub_menu == "Generate Model":
#			st.warning("under development")
			TrainModel.createPage()
		elif sub_menu == "Apply Model":
#			st.warning("under development")
			TestModel.createPage()
	elif main_menu == "Option":
			st.warning("under development")


#Run the code
def main():
	if not firebase_admin._apps:
		cred = firebase_admin.credentials.Certificate(st.secrets["firebase"])
		firebase_admin.initialize_app(cred)
		st.session_state.db = firebase_admin.firestore.client()
	st.session_state.authenticator = OpenAuthenticator()
	if "authentication_status" not in st.session_state:
		st.session_state.authentication_status = None
	if "username" not in st.session_state:
		st.session_state.username = None
	if "name" not in st.session_state:
		st.session_state.name = None
	
	if st.session_state.authentication_status:
		show_main_app()
	else:
		show_login_page()

if __name__ == "__main__":
	main()
