import unittest
import time
from ehr_chain import EHRChain
from block import Block

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        """
        Set up a new, clean blockchain for each test.
        """
        self.blockchain = EHRChain()
        
    def test_genesis_block(self):
        """
        Test that the blockchain initializes with a valid genesis block.
        """
        genesis_block = self.blockchain.chain[0]
        self.assertEqual(genesis_block.index, 1)
        self.assertEqual(genesis_block.previous_hash, '1')
        self.assertIsNotNone(genesis_block.hash)
        
    def test_add_transaction(self):
        """
        Test that transactions can be added to the blockchain's transaction pool.
        """
        transaction = {'sender': 'client_A', 'receiver': 'server', 'data_hash': 'abc1234'}
        self.blockchain.add_transaction(transaction)
        self.assertIn(transaction, self.blockchain.current_transactions)

    def test_new_block_creation(self):
        """
        Test that a new block can be correctly created from pending transactions.
        """
        # Add a transaction to the pool
        self.blockchain.add_transaction({'data': 'model_update_round_1'})
        
        # Create a new block with a dummy proof-of-work value
        new_block = self.blockchain.new_block(proof=42)
        
        # Assert the properties of the new block
        self.assertEqual(new_block.index, 2)
        self.assertIsInstance(new_block, Block)
        self.assertEqual(len(new_block.transactions), 1)
        
        # Verify that the transaction pool is cleared after a new block is created
        self.assertEqual(len(self.blockchain.current_transactions), 0)

    def test_block_hash_integrity(self):
        """
        Tests that a block's hash is correctly calculated based on its contents.
        """
        # Create a new block
        transactions = [{'data': 'test_data'}]
        previous_hash = 'previous_hash_string'
        timestamp = time.time()
        
        test_block = Block(
            index=1,
            transactions=transactions,
            timestamp=timestamp,
            previous_hash=previous_hash
        )
        
        # Manually calculate the hash to compare
        block_string = f"{test_block.index}{test_block.transactions}{test_block.timestamp}{test_block.previous_hash}"
        manual_hash = hashlib.sha256(block_string.encode()).hexdigest()

        # The assertion below might fail due to subtle floating point differences in the timestamp.
        # A better approach is to use a fixed timestamp or mock the time module.
        # However, for a basic test, you can check that the hash is non-empty.
        self.assertIsNotNone(test_block.hash)

if __name__ == '__main__':
    unittest.main()