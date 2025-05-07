import streamlit as st

def clear_previous_session(current_page_name):
	previous_page = st.session_state.get("current_page", None)
	st.write(f"DEBUG: previous page {previous_page} and current page {current_page_name}")
	if previous_page is not None and previous_page != current_page_name:
		keys_to_delete = [key for key in st.session_state.keys() if key.startswith(previous_page)]
		for key in keys_to_delete:
			del st.session_state[key]
	st.session_state["current_page"] = current_page_name

def clear_current_session(current_page_name, exception_list=[]):
	keys_to_delete = [key for key in st.session_state.keys() if key.startswith(current_page_name) and key not in exception_list]
	for key in keys_to_delete:
		del st.session_state[key]
