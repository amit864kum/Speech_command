import concurrent.futures
import random
import torch
import time
from typing import Dict, Tuple, List
import os.path

from miner import Miner
from ehr_chain import EHRChain
from p2p_sim import P2PNode
from block import Block

# FL imports
from data_loader import SpeechCommandsDataLoader
from fl_node import FLNode
from fl_trainer import aggregate_local_models, compute_global_model_hash

# ----------------------------------------------
# Configuration
# ----------------------------------------------
NUM_SAMPLES_PER_EPOCH = 15
EPOCHS = 4
MINERS = ["Device_A", "Device_B", "SmartSpeaker_C"]

# FL hyperparameters
LOCAL_EPOCHS = 4
BATCH_SIZE = 32
LR = 1e-3

# GKWS dataset params
NUM_CLASSES = 10
INPUT_DIM = 64

# ----------------------------------------------
# Helper for mining
# ----------------------------------------------
def mine_block(miner: Miner, model_hash: str, records: list):
    return miner.mine_block(records, model_hash=model_hash)

# ----------------------------------------------
# Setup blockchain and miners
# ----------------------------------------------
blockchain_filepath = "blockchain.json"
if os.path.exists(blockchain_filepath):
    ehr_chain = EHRChain.load_from_file(filepath=blockchain_filepath)
else:
    ehr_chain = EHRChain()
    print(f"[📂] New blockchain created with {len(ehr_chain.chain)} blocks.")

miners = [Miner(miner_id=m, chain=ehr_chain, difficulty=2) for m in MINERS]
p2p_nodes = {m.miner_id: P2PNode(m.miner_id, ehr_chain) for m in miners}

print(f"[📌] Canonical chain tip: #{ehr_chain.chain[-1].index} | Hash: {ehr_chain.chain[-1].hash[:12]}...")

# ----------------------------------------------
# Prepare FL node datasets and FLNode objects
# ----------------------------------------------
print("\n[⏳] Initializing and downloading Speech Commands dataset...")
data_loader = SpeechCommandsDataLoader(num_clients=len(MINERS))

fl_nodes: Dict[str, FLNode] = {}
for i, miner_id in enumerate(MINERS):
    print(f"[{i+1}/{len(MINERS)}] Initializing FLNode for {miner_id}...")
    
    start_time = time.time()
    client_data = data_loader.get_client_data(client_id=i)
    data_loading_latency = time.time() - start_time
    
    fl_node = FLNode(
        node_id=miner_id,
        local_data=client_data,
        input_dim=INPUT_DIM,
        output_dim=NUM_CLASSES,
        device="cpu"
    )
    fl_nodes[miner_id] = fl_node
    print(f"    -> Node {miner_id} has {len(client_data)} samples. Data loading latency: {data_loading_latency:.2f}s")

# ----------------------------------------------
# Helper: Assign random samples
# ----------------------------------------------
def assign_samples(epoch: int) -> Dict[str, str]:
    samples = [f"Sample_{epoch:03}{i:02}" for i in range(NUM_SAMPLES_PER_EPOCH)]
    assignments: Dict[str, str] = {}
    for s in samples:
        miner_id = random.choice(MINERS)
        assignments[s] = miner_id
    return assignments

# ----------------------------------------------
# Federated training + mining loop (multi-round FedAvg)
# ----------------------------------------------
for epoch in range(1, EPOCHS + 1):
    print(f"\n====================== EPOCH {epoch} ======================")
    
    # Step 1: Local training
    local_weights: Dict[str, Dict] = {}
    local_accuracies: Dict[str, float] = {}
    local_model_hashes: Dict[str, str] = {}
    local_predictions: Dict[str, List] = {}
    
    for miner_id, node in fl_nodes.items():
        print(f"\n[🏥] Node {miner_id} starting local training...")
        
        start_time = time.time()
        weights_serialized, acc, model_id, predictions = node.local_train(
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR
        )
        local_training_latency = time.time() - start_time
        
        local_weights[miner_id] = weights_serialized
        local_accuracies[miner_id] = acc * 100.0
        local_model_hashes[miner_id] = model_id
        local_predictions[miner_id] = predictions
        
        print(f"[🏥] Node {miner_id} finished. Local acc: {acc*100:.2f}%, training latency: {local_training_latency:.2f}s")
        print(f"    - Local Model Hash: {model_id}")

    print("\nLocal Accuracies (per-node):")
    for mid, acc in local_accuracies.items():
        print(f"  - {mid}: {acc:.2f}%")

    # Step 2: Aggregate (FedAvg) -> produce global_weights
    print("\n[🔗] Aggregating local models with FedAvg...")
    agg_start_time = time.time()
    try:
        global_weights = aggregate_local_models(local_weights)
    except Exception as e:
        print(f"[❌] Aggregation failed: {e}")
        continue
    agg_latency = time.time() - agg_start_time

    global_hash = compute_global_model_hash(global_weights)
    print(f"[🔐] Global model computed. Hash: {global_hash}. Aggregation latency: {agg_latency:.2f}s")

    # Step 3: Update local nodes with the global model
    print("[🔄] Updating local nodes with aggregated global weights...")
    for miner_id, node in fl_nodes.items():
        node.update_model(global_weights)

    # Step 4: Mine a block that references the global model hash
    txs = []
    for miner_id, preds in local_predictions.items():
        record = {
            "miner": miner_id,
            "predictions": preds
        }
        txs.append(record)
    
    print("\n[🔁] Starting PoW race for the global model...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(mine_block, miner_obj, global_hash, txs): miner_obj.miner_id for miner_obj in miners}
        
        winner_block = None
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                if res:
                    winner_block = res
                    break
            except Exception as e:
                print(f"[❌] Miner task raised exception: {e}")
            
    if not winner_block:
        print("[❌] No miner found a valid block this round.")
        continue
    
    print(f"[➡️] {winner_block.miner} broadcasting Block #{winner_block.index}...")
    
    if ehr_chain.add_block(winner_block):
        print(f"[✔️] Block #{winner_block.index} appended to main chain.")
    else:
        print(f"[❌] Block #{winner_block.index} rejected by main chain.")
        
    for node_id, node in p2p_nodes.items():
        if node_id != winner_block.miner:
            accepted = node.receive_block(winner_block)
            if accepted:
                print(f"[✔️] Node {node_id} accepted Block #{winner_block.index}")
            else:
                print(f"[❌] Node {node_id} rejected Block #{winner_block.index}")
    
    print("🔎 --- Block Details ---")
    print(f"🔢 Index: {winner_block.index}")
    print(f"🕒 Timestamp: {time.ctime(winner_block.timestamp)}")
    print(f"🔗 Previous Hash: {winner_block.prev_hash[:12]}...")
    print(f"💻 Current Hash: {winner_block.hash[:12]}...")
    print(f"🎛️ Nonce: {winner_block.nonce}")
    print(f"🧑‍⚕️ Mined by: {winner_block.miner}")
    print(f"🌐 Global Model Hash: {winner_block.model_hash[:12]}...")
    print(f"📜 Digital Signature: {winner_block.signature[:12]}...")
    print("📦 Transactions:")
    for record in winner_block.records:
        print(f"    - Miner: {record['miner']}")
        for pred in record['predictions']:
            print(f"        -> Prediction: '{pred['predicted_word']}' (Confidence: {pred['confidence']:.2f}%)")
    print("🔎 ---------------------")

print("\n[✓] Federated training + blockchain mining simulation complete.")

ehr_chain.save_to_file()
