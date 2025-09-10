# ehr_chain.py
import hashlib
import json
import time
from typing import List, Dict

from block import Block

class EHRChain:
    def __init__(self):
        self.chain = []
        self.difficulty = 2

    def create_genesis_block(self):
        genesis_block = Block(
            index=0,
            prev_hash="0",
            records=[],
            access_logs=[],
            miner="Genesis",
            model_hash="0",
            difficulty=self.difficulty
        )
        self.chain.append(genesis_block)
    
    @classmethod
    def load_from_file(cls, filepath="blockchain.json"):
        """Loads a blockchain from a JSON file."""
        ehr_chain = cls()
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if 'chain' in data:
                    ehr_chain.chain.clear()
                    for block_data in data['chain']:
                        block = Block(
                            index=block_data['index'],
                            prev_hash=block_data['prev_hash'],
                            records=block_data['records'],
                            access_logs=block_data.get('access_logs', []),
                            miner=block_data['miner'],
                            model_hash=block_data['model_hash'],
                            difficulty=block_data.get('difficulty', ehr_chain.difficulty),
                            # Corrected: Read the 'proof' key from the file and pass it as the 'nonce'
                            nonce=block_data.get('proof', None) 
                        )
                        block.hash = block_data['hash']
                        ehr_chain.chain.append(block)
                print(f"[📂] Loaded blockchain with {len(ehr_chain.chain)} blocks from {filepath}.")
                return ehr_chain
        except FileNotFoundError:
            print(f"[❌] {filepath} not found. Creating a new blockchain.")
            ehr_chain.create_genesis_block()
            return ehr_chain

    def add_block(self, block):
        if not self.is_valid_block(block):
            return False
        
        self.chain.append(block)
        return True

    def is_valid_block(self, block):
        last_block = self.chain[-1]
        if block.prev_hash != last_block.hash:
            print("Block rejected: Previous hash is invalid.")
            return False
        
        if not block.hash.startswith("0" * self.difficulty):
            print("Block rejected: Proof of Work is invalid.")
            return False
        
        return True

    def save_to_file(self, filepath="blockchain.json"):
        """Saves the blockchain to a JSON file."""
        chain_list = []
        for block in self.chain:
            block_dict = {
                "index": block.index,
                "timestamp": block.timestamp,
                "records": block.records,
                "proof": block.nonce,
                "prev_hash": block.prev_hash,
                "model_hash": block.model_hash,
                "hash": block.hash,
                "miner": block.miner,
                "signature": getattr(block, 'signature', None),
                "public_key": getattr(block, 'public_key', None),
                "difficulty": block.difficulty
            }
            chain_list.append(block_dict)

        data_to_save = {"chain": chain_list}
        with open(filepath, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"[💾] Blockchain saved to {filepath}")
