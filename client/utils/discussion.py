import streamlit as st
from datetime import datetime
from server.data.firestore_handler import CommentContainer, save_CC, load_CCs
from server.data.backblaze_handler import upload_discussion_pdf, generate_presigned_url

username = st.session_state.get("username")
data_name = "tmp.dat"
def commentlist(data_id: str):
	# Comment list
	load_CCs(data_id)
	for CC in load_CCs(data_id):
		st.markdown(f"**#{CC.id}** {CC.username} {CC.timestamp}")
		st.markdown(CC.content)
		if CC.pdf_flag:
			if st.button(CC.pdf_filename, key=CC.id):
				pdf_url = generate_presigned_url(CC.pdf_filename)
				js = f'<script> window.open("{pdf_url}", "_blank"); </script>'
				st.components.v1.html(js)
		st.markdown("---")
	
	with st.form("comment_form", clear_on_submit=True):
		content = st.text_area("Write your opinion", height=100, max_chars=1000, )
		uploaded_pdf = st.file_uploader("Attach PDF (optional)", type=["pdf"])
		submitted = st.form_submit_button("Add comment")
	
	if submitted:
		if not content:
			st.error("Put a content message to proceed")
		else:
			if uploaded_pdf:
				new_Cpdf_id = upload_discussion_pdf(data_id, uploaded_pdf)
				pdf_filename = f"{data_id}-discussion_{new_Cpdf_id}.pdf"
			else:
				pdf_filename = ""
			save_CC(data_id, content, pdf_filename)
			st.success("Your comment has successfully been uploaded on the discussion")
