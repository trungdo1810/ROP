import yaml

# Hàm đọc cấu hình từ file YAML
def load_config(config_file='ROP-train/configs/config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Đọc các cấu hình miner và loss từ config.yaml
batch_easy_hard_miner_options = config['mining']['batch_easy_hard_miner']
triplet_margin_miner_options = config['mining']['triplet_margin_miner']
loss_options = config['loss']

# Loop qua các tổ hợp
for model_name, mining_strategy, loss_name, miner_config in itertools.product(
        config['model']['types'],  # 3 models
        ['batch_easy_hard_miner', 'triplet_margin_miner'],  # 2 mining strategies
        loss_options.keys(),  # 2 loss functions
        batch_easy_hard_miner_options if mining_strategy == 'batch_easy_hard_miner' else triplet_margin_miner_options):

    print(f"Training model: {model_name}, Mining strategy: {mining_strategy}, Loss function: {loss_name}, Config: {miner_config}")
    # Các bước huấn luyện
