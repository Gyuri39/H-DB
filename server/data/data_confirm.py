import streamlit as st
from datetime import datetime, timezone, timedelta
from firebase_admin import firestore 

def iso_kst_now():
	return (datetime.now(timezone.utc) + timedelta(hours=9)).isoformat()

def update_verification(doc_id, entry):
	db_client = firestore.client()
	collection_ref = db_client.collection('datasets')
	doc_ref = collection_ref.document(doc_id)
	transaction = db_client.transaction()

	@firestore.transactional
	def txn_op(tx, ref, entry):
		snap = ref.get(transaction=tx)
		curr = snap.to_dict() or {}
		who = curr.get("who_verified", {})

		now = iso_kst_now()
		who.update({
			"status": entry["status"],
			"by_name": entry["by_name"],
			"at": entry.get("at",now),
			"note": entry.get("note",""),
		})

		hist = who.get("history", [])
		if not isinstance(hist, list):
			hist = []
		hist.append({
			k: who[k] for k in [
				"status", "by_name", "at", "note"
			]
		})
		who["history"] = hist
		tx.update(ref, {"who_verified": who})

	txn_op(transaction, doc_ref, entry)

