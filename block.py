# block.py
import hashlib
import json
import time

class Block:
    def __init__(self, index, prev_hash, records, access_logs, miner, model_hash, difficulty, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.prev_hash = prev_hash
        self.records = records
        self.access_logs = access_logs  # For storing signatures and access info
        self.miner = miner
        self.model_hash = model_hash
        self.difficulty = difficulty
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        # We don't include the signature in the hash, as it's added after PoW
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "records": self.records,
            "miner": self.miner,
            "model_hash": self.model_hash,
            "difficulty": self.difficulty,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()