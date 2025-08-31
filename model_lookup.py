import os
import json
import sys

def lookup_model_on_chain(model_hash):
    # Ensure the blockchain file exists
    blockchain_file = "blockchain.json"
    if not os.path.exists(blockchain_file):
        print(f"[❌] Error: Blockchain file '{blockchain_file}' not found. Please run ehr_main.py first to create the blockchain.")
        return

    # Load the blockchain from the file
    with open(blockchain_file, "r") as f:
        chain_data = json.load(f)

    print(f"\n🔍 Searching blockchain for model hash: {model_hash}")
    print("=============================================")

    found = False
    for block in chain_data["chain"]:
        if "model_id" in block["data"] and block["data"]["model_id"] == model_hash:
            found = True
            print(f"✅ Found model hash in Block #{block['index']}!")
            print(f"  - Miner Node    : {block['miner']}")
            print(f"  - Previous Hash : {block['prev_hash'][:10]}...")
            print(f"  - Timestamp     : {block['timestamp']}")
            print(f"  - Proof of Work : {block['proof']}")
            print(f"  - Block Hash    : {block['hash']}")
            print("\n  --- Model Record Details ---")
            print(f"  - Accuracy      : {block['data']['accuracy']:.4f}")
            print(f"  - Epochs        : {block['data']['epochs']}")
            print(f"  - LR            : {block['data']['lr']}")
            print(f"  - Batch Size    : {block['data']['batch_size']}")
            print("\n=============================================")
            break

    if not found:
        print(f"[⚠️] Model hash not found on the blockchain.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_lookup.py <model_hash>")
        sys.exit(1)

    input_hash = sys.argv[1]
    lookup_model_on_chain(input_hash)