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
        self.private_key = f"privkey-{miner_id}"
        self.public_key = f"pubkey-{miner_id}"

    def sign_block(self, block_hash: str):
        return hashlib.sha256((self.private_key + block_hash).encode()).hexdigest()

    def mine_block(self, records: List[Dict], model_hash: str):
        # We replace PoW with a simple, quick block signing for this PoA-like model
        last_block = self.chain.chain[-1]
        new_index = last_block.index + 1
        prev_hash = last_block.hash

        new_block = Block(
            index=new_index,
            prev_hash=prev_hash,
            records=records,
            access_logs=[],
            miner=self.miner_id,
            model_hash=model_hash,
            difficulty=0, # Simplified PoA consensus
            nonce=0
        )
        
        new_block.signature = self.sign_block(new_block.hash)
        new_block.public_key = self.public_key

        print(f"[⛏️] Miner {self.miner_id} signed block #{new_block.index}.")
        return new_block
