import os
import json
from crypto_utils import hash_record, sign_record
import time

EHR_DIR = "sample_records"
os.makedirs(EHR_DIR, exist_ok=True)

# Fake upload API

def store_ehr(patient_id, record_type, content):
    record = {
        "patient_id": patient_id,
        "record_type": record_type,
        "timestamp": time.time(),
        "content": content
    }
    filename = f"{EHR_DIR}/{patient_id}_{record_type}.json"
    with open(filename, 'w') as f:
        json.dump(record, f)

    record_hash = hash_record(record)
    signature = sign_record(record_hash)
    return filename, record_hash, signature