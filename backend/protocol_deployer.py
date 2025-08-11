def deploy_protocol(name="MysticVault"):
    print(f"[PROTOCOL] Deploying protocol: {name}")
    # Simulated deployment
    open(f"{name}_spec.md", "w").write(f"# {name}\nAuto-deployed by AI.")
    print(f"[PROTOCOL] {name}_spec.md created.")
