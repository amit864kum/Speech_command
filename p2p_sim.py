from ehr_chain import EHRChain

# ---------------- p2p_sim.py ----------------
class P2PNode:
    def __init__(self, miner_id, chain):
        self.miner_id = miner_id
        self.chain = chain

    def receive_block(self, block):
        """
        Simulate receiving a block from another miner.
        Add to chain if valid.
        """
        if self.chain.add_block(block):
            print(f"[✔️] Node {self.miner_id} accepted Block #{block.index} (🌐 {block.model_hash})")
            return True
        else:
            print(f"[❌] Node {self.miner_id} rejected Block #{block.index}")
            return False
