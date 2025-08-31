import hashlib
import rsa
import json

# Simple RSA keypair simulation
(key_pub, key_priv) = rsa.newkeys(512)

def hash_record(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

def sign_record(record_hash):
    return rsa.sign(record_hash.encode(), key_priv, 'SHA-256')

def verify_signature(record_hash, signature):
    try:
        rsa.verify(record_hash.encode(), signature, key_pub)
        return True
    except:
        return False