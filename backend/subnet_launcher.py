def build_subnet_config(chain_name="MysticSubnet", validators=5):
    config = {
        "chain": chain_name,
        "consensus": "PoS",
        "validators": validators,
        "ai_controlled": True,
    }
    with open("subnet_config.json", "w") as f:
        import json

        json.dump(config, f, indent=2)
    print(f"[SUBNET] Config saved for {chain_name}")
