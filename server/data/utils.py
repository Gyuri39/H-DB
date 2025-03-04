import pandas as pd
import numpy as np
from thefuzz import process
from client.utils import KEY_PROPERTY

def candidate_columns(df: pd.DataFrame, key_cols: list, threshold=80):
	"""
	Figure out candidate column names that are similar to key_cols.
	If a change is recommended, one can utilize the suggested dataframe with changed column names for unity.
	"""
	mapping = {}
	to_suggest = False
	col_list = list(df.columns)
	matched_cols = set()
	
	# Firstly convert 'TEMP' column
	if 'TEMP' in key_cols:
		match, score = process.extractOne('TEMP', col_list) if col_list else(None, 0)
		if score >= threshold:
			mapping[match] = 'TEMP'
			col_list.remove(match)
			matched_cols.add('TEMP')
			if match != 'TEMP':
				to_suggest = True
		key_cols_tmp = [item for item in key_cols if item is not 'TEMP']
	else:
		key_cols_tmp = key_cols
	
	# Remove matching to unwanted KEY features
	unwanted_colnames = [item for item in KEY_PROPERTY if item not in key_cols]
	col_list = [item for item in col_list if item not in unwanted_colnames]


	# Convert the other rows
	for colname in key_cols_tmp:
		match, score = process.extractOne(colname, col_list) if col_list else (None, 0)
		if score >= threshold:
			mapping[match] = colname
			col_list.remove(match)
			matched_cols.add(colname)
			if match != colname:
				to_suggest = True
	
	# Mapping to key_cols
	df_map = df.rename(columns = mapping)
	df_map = df_map[[col for col in key_cols if col in df_map.columns]]

	if to_suggest or len(matched_cols) != len(list(df.columns)):
		return True, df_map
	else:
		return False, df_map


def match_columns(df1, df2, key_col=None, threshold=80):
	"""
	Match similar column names b/w two dataframes using fuzzy matching.
	Returns a mapping of df2 columns to df1 columns.
	Assumes key_col is included in both df1 and df2, especially exactly in df1.
	"""
	mapping = {}
	df1columns = df1.columns
	df2columns = df2.columns
	# match key_col firstly
	if key_col != None:
		match, score = process.extractOne(key_col, df2columns)
		if score >= threshold:
			mapping[match] = key_col
			df1columns.remove(key_col)
			df2columns.remove(match)
	# match all df2.columns
	for col1 in df1columns:
		match, score = process.extractOne(col1, df2columns)
		if score >= threshold:
			mapping[match] = col1
	return mapping


def number2string(num):
	"""
	Convert number to a standardized string format for typo detection.
	"""
	return "{:.8g".format(num).replace("e+0", "e").replace("e-0", "e")


def is_single_digit_typo(num1, num2):
	"""
	Detects if two numbers differ by only one digit (substitution, insertion, or deletion)
	Is to be applied provided num1 != num2.
	"""
	str1, str2 = number2string(num1), number2string(num2)

	# Case 0: more than a single digit difference
	if abs(len(str1) - len(str2)) > 1:
		return False
	
	# Case 1: same or substitution (e.g. 3.141592 -> 4.141592)
	if len(str1) == len(str2):
		mismatch_count = sum(d1 != d2 for d1, d2 in zip(str1, str2))
		if mismatch_count <= 1:
			return True
	
	# Case 2: insertion/deletion (e.g. 3.141592 -> 3.1415992)
	if abs(len(str1) - len(str2)) == 1:
		for i in range(min(len(str1), len(str2))):
			if str1[:i] + str1[i+1:] == str2 or str2[:i] + str2[i+1:] == str1:
				return True

	return False


def align_dataframes(df1, df2, key_col): # -> merged dataframe, common single column which should be accurate
	"""
	Align two dataframes by common columns and key columns.
	Merged dataframe as well as common columns list are returned.
	The key_col is shared while the other common columns are printied with suffixes "_1" and "_2".
	"""
	# Attempt to standardize column names using fuzzy matching
	col_mapping = match_columns(df1, df2)
	df2 = df2.rename(columns = col_mapping)
	common_cols = sorted(set(df1.columns) & set(df2.columns))
	if not common_cols:
		return None, None
	else:
		# Convert to float
		df1_common = df1[common_cols].astype(float)
		df2_common = df2[common_cols].astype(float)
		if key_col in common_cols:
			common_other_cols = [col for col in common_cols if col != key_col]
			common_cols = [key_col] + common_other_cols
		df1_sorted = df1_common.sort_values(by=common_cols).reset_index(drop=True)
		df2_sorted = df2_common.sort_values(by=common_cols).reset_index(drop=True)
		merged = df1_common.merge(df2_common, on=key_col, how="outer", suffixes=("_1", "_2"))
##		merged = merged.sort_values(by=###############################
		return merged, common_cols


def detect_digit_typos(df, common_cols, key_col):	# df = merged df
	"""
	Identifies typos where a single digit differs
	"""
	for col in common_cols:
		col_1, col_2 = f"{col}_1", f"{col}_2"
		if col_1 in df.columns and col_2 in df.columns:
			typos = df[df.apply(lambda row: is_single_digit_typo(row[col_1], row[col_2]), axis=1)]
			if not typos.empty:
				print(f"Possible single-digit typo dected in '{col}':\n{typos[[col_1, col_2]]}\n")
		else:
			raise ValueError("Input common_cols are not a common col of corresponding merged dataframe.")


def are_dataframes_similar(df1, df2, key_col='temp', tolerance=0.05):
	"""
	Advanced tool to compare two dataframes considering up to single digit typos.
	No typos are assumed in key_col and matching up to 'tolerance' relative tolerance.
	"""
	df_merged, common_cols = align_dataframes(df1, df2, key_col)
	if df_merged is None:
		return False
	


def are_dataframes_equal(df1, df2, key_col='temp', tolerance=0.05):
	"""
	Simple tool to compare two dataframes matching up to 'tolerance' relative tolerance.
	"""
	common_cols = sorted(set(df1.columns) & set(df2.columns))
	if key_col in common_cols:
		common_other_cols = [col for col in common_cols if col != key_col]
		common_cols = [key_col] + common_other_cols

	df1_common = df1[common_cols].astype(float)
	df2_common = df2[common_cols].astype(float)
	df1_sorted = df1_common.sort_values(by=common_cols).reset_index(drop=True)
	df2_sorted = df2_common.sort_values(by=common_cols).reset_index(drop=True)
	if df1_sorted.shape != df2_sorted.shape:
		return False
	return np.allclose(df1_sorted, df2_sorted, rtol=tolerance, atol=0)
