# miner.py
import hashlib
import time
from typing import Dict, List

from block import Block
from ehr_chain import EHRChain

class Miner:
    def __init__(self, miner_id, chain, difficulty):
        self.miner_id = miner_id
        self.chain = chain
        self.difficulty = difficulty
        # Simulate keys (in real systems, use proper crypto keys)
        self.private_key = f"privkey-{miner_id}"
        self.public_key = f"pubkey-{miner_id}"

    def sign_block(self, block_hash: str):
        # A mock digital signature: hash(private_key + block_hash)
        return hashlib.sha256((self.private_key + block_hash).encode()).hexdigest()

    def mine_block(self, records: List[str], model_hash: str):
        last_block = self.chain.chain[-1]
        new_index = last_block.index + 1
        prev_hash = last_block.hash

        # Create the new block object
        new_block = Block(
            index=new_index,
            prev_hash=prev_hash,
            records=records,
            access_logs=[],
            miner=self.miner_id,
            model_hash=model_hash,
            difficulty=self.difficulty
        )

        prefix = "0" * self.difficulty
        nonce = 0
        start_time = time.time()
        while True:
            new_block.nonce = nonce
            candidate_hash = new_block.compute_hash()
            if candidate_hash.startswith(prefix):
                new_block.hash = candidate_hash
                break
            nonce += 1
        
        # Sign with miner’s private key
        new_block.signature = self.sign_block(new_block.hash)
        new_block.public_key = self.public_key

        print(f"[⛏️] Miner {self.miner_id} mined block #{new_block.index} in {round(time.time()-start_time,2)}s.")
        return new_block