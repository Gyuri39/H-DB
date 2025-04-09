import boto3
import streamlit as st
from botocore.config import Config
from botocore.exceptions import ClientError
import re

def get_b2_client():
	b2 = st.secrets["b2"]
	b2_client = boto3.client(service_name='s3',
						endpoint_url=b2["end_point"],
						aws_access_key_id=b2["key_id"],
						aws_secret_access_key=b2["application_key"]
				)
	return b2_client

def get_b2_resource():
	b2=st.secrets["b2"]
	b2_resource = boto3.resource(service_name='s3',
						endpoint_url=b2["end_point"],
						aws_access_key_id=b2["key_id"],
						aws_secret_access_key=b2["application_key"],
						config=Config(
							signature_version='s3v4',
						))
	return b2_resource

def upload_pdf(file, key):
	b2 = st.secrets["b2"]
	b2_rw = get_b2_resource()
	try:
		response = b2_rw.Bucket(b2["bucket_name"]).upload_file(file, key+'.pdf')
	#	config = boto3.s3.transfer.TransferConfig(use_threads=False)
	#	response = b2_rw.Bucket(b2["bucket_name"]).upload_file(file, key+'.pdf', ExtraArgs={'ContentType': 'application/pdf'}, Config=config)
		return response
	except ClientError as ce:
		print('error', ce)

def generate_presigned_url(key, expiration_seconds = 3600):
	try:
		b2 = st.secrets["b2"]
		b2_private = get_b2_resource()
		response = b2_private.meta.client.generate_presigned_url(
						ClientMethod = 'get_object',
						ExpiresIn = expiration_seconds,
						Params = {
							'Bucket': b2["bucket_name"],
							'Key': key
						}
					)
		return response
	except ClientError as ce:
		print('error', ce)

def upload_discussion_pdf(data_id:str, uploaded_pdf) -> int:
	b2 = st.secrets["b2"]
	b2_rw = get_b2_resource()
	bucket = b2_rw.Bucket(b2["bucket_name"])
	pattern = re.compile(f"^{re.escape(data_id)}-discussion_(\\d+)\\.pdf$")

	new_Cpdf_id = 1
	for obj in bucket.objects.all():
		match = pattern.match(obj.key)
		if match:
			new_Cpdf_id += 1

	new_filename = f"{data_id}-discussion_{new_Cpdf_id}.pdf"
	try:
		bucket.upload_fileobj(uploaded_pdf, new_filename)
	except ClientError as ce:
		st.error(f"File upload error: {ce}")
		raise ce

	return new_Cpdf_id
