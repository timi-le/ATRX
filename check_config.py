import yaml

# Load and display config
with open("services/execution/config_mt5.yaml") as f:
    config = yaml.safe_load(f)

mt5_config = config["mt5"]
print(f"Login: {mt5_config['login']}")
print(f"Password: {mt5_config['password']}")
print(f"Server: {mt5_config['server']}")
print(f"Magic Number: {mt5_config['magic_number']}")
