import torch
from torch.utils.data import DataLoader
from dataset import ItemData, SeqData, RecDataset

# ✅ 设置根目录（你已经下载并处理的地方）
AMAZON_ROOT = "dataset/amazon/raw"
ML32M_ROOT = "dataset/ml-32m/raw"


def run_amazon():
    print(">>> Loading Amazon Beauty Data...")

    # 加载 item-level 特征（用于冷启动、embedding 学习等）
    item_data = ItemData(
        root=AMAZON_ROOT,
        dataset=RecDataset.AMAZON,
        split="beauty",
        force_process=True,
        train_test_split="train"
    )

    # 加载用户行为序列数据（用于训练序列模型）
    seq_data = SeqData(
        root=AMAZON_ROOT,
        dataset=RecDataset.AMAZON,
        split="beauty",
        is_train=True,
        force_process=True
    )

    item_loader = DataLoader(item_data, batch_size=32, shuffle=True)
    seq_loader = DataLoader(seq_data, batch_size=8, shuffle=True)

    # 取一批数据
    for batch in seq_loader:
        print("Amazon Sequence batch:")
        print(f"user_ids: {batch.user_ids.shape}")
        print(f"ids: {batch.ids.shape}")
        print(f"x: {batch.x.shape}")
        break


def run_ml32m():
    print(">>> Loading MovieLens-32M Data...")

    item_data = ItemData(
        root=ML32M_ROOT,
        dataset=RecDataset.ML_32M,
        force_process=True,
        train_test_split="train"
    )

    seq_data = SeqData(
        root=ML32M_ROOT,
        dataset=RecDataset.ML_32M,
        is_train=True,
        force_process=False
    )

    item_loader = DataLoader(item_data, batch_size=32, shuffle=True)
    seq_loader = DataLoader(seq_data, batch_size=8, shuffle=True)

    for batch in seq_loader:
        print("ML-32M Sequence batch:")
        print(f"user_ids: {batch.user_ids.shape}")
        print(f"ids: {batch.ids.shape}")
        print(f"x: {batch.x.shape}")
        break


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 运行 Amazon Beauty 数据集
    run_amazon()

    print("=" * 40)

    # 运行 MovieLens 32M 数据集
    run_ml32m()