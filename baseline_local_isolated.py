"""baseline_local_isolated.py - Isolated client evaluation for Table 1"""
import requests

def run():
    clients = [8001, 8002, 8003]
    drug_id = "CID000000271"
    
    for port in clients:
        try:
            r = requests.get(
                f"http://localhost:{port}/retrieve", 
                params={"drug_id": drug_id, "include_weights": "false"},
                timeout=120
            )
            r.raise_for_status()
            data = r.json()
            print(f"Client {port} Metrics:", data.get("metrics"))
            print(f"Client {port} Confidence:", data.get("local_confidence"))
        except Exception as e:
            print(f"Client {port} failed: {e}")

if __name__ == "__main__":
    run()
