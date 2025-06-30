#!/usr/bin/env python3
"""
数据集形状检查工具
用于检查Amazon和ML数据集的所有数据形状和统计信息
"""

import os
import sys
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.processed import ItemData, SeqData, RecDataset
from data.schemas import SeqBatch
from torch.utils.data import DataLoader


def print_separator(title: str, char: str = "=", width: int = 80):
    """打印分隔线"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_subsection(title: str, char: str = "-", width: int = 60):
    """打印子标题"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def examine_tensor_stats(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """检查张量的详细统计信息"""
    stats = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'numel': tensor.numel(),
        'memory_mb': tensor.numel() * tensor.element_size() / 1024 / 1024,
    }
    
    if tensor.dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        try:
            stats.update({
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'mean': tensor.float().mean().item(),
                'std': tensor.float().std().item(),
            })
            
            # 检查特殊值
            if tensor.dtype in [torch.int32, torch.int64]:
                unique_vals = torch.unique(tensor)
                stats['unique_count'] = len(unique_vals)
                stats['has_negative_one'] = (-1 in unique_vals).item() if len(unique_vals) > 0 else False
                
                # 计算填充比例（假设-1是填充值）
                if tensor.numel() > 0:
                    padding_count = (tensor == -1).sum().item()
                    stats['padding_ratio'] = padding_count / tensor.numel()
                else:
                    stats['padding_ratio'] = 0.0
            
        except Exception as e:
            stats['stats_error'] = str(e)
    
    return stats


def examine_item_dataset(dataset_name: str, dataset_type: RecDataset, 
                        dataset_folder: str, split: str = None) -> Dict[str, Any]:
    """检查ItemData数据集"""
    print_subsection(f"ItemData - {dataset_name}")
    
    try:
        # 加载数据集
        if split:
            item_dataset = ItemData(
                root=dataset_folder,
                dataset=dataset_type,
                force_process=False,
                split=split
            )
        else:
            item_dataset = ItemData(
                root=dataset_folder,
                dataset=dataset_type,
                force_process=False
            )
        
        print(f"数据集长度: {len(item_dataset)}")
        
        # 检查单个样本
        sample = item_dataset[0]
        print(f"样本类型: {type(sample)}")
        
        stats = {}
        for field_name in sample._fields:
            field_value = getattr(sample, field_name)
            if isinstance(field_value, torch.Tensor):
                field_stats = examine_tensor_stats(field_value, field_name)
                stats[field_name] = field_stats
                
                print(f"  {field_name}:")
                print(f"    形状: {field_stats['shape']}")
                print(f"    数据类型: {field_stats['dtype']}")
                print(f"    内存: {field_stats['memory_mb']:.2f} MB")
                if 'min' in field_stats:
                    print(f"    值范围: [{field_stats['min']}, {field_stats['max']}]")
                if 'unique_count' in field_stats:
                    print(f"    唯一值数量: {field_stats['unique_count']}")
        
        # 检查batch
        dataloader = DataLoader(item_dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"\n  批次形状 (batch_size=4):")
        for field_name in batch._fields:
            field_value = getattr(batch, field_name)
            if isinstance(field_value, torch.Tensor):
                print(f"    {field_name}: {field_value.shape}")
        
        return {
            'dataset_length': len(item_dataset),
            'sample_stats': stats,
            'batch_shapes': {field: getattr(batch, field).shape for field in batch._fields 
                           if isinstance(getattr(batch, field), torch.Tensor)}
        }
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def examine_seq_dataset(dataset_name: str, dataset_type: RecDataset, 
                       dataset_folder: str, is_train: bool, split: str = None) -> Dict[str, Any]:
    """检查SeqData数据集"""
    split_name = "训练" if is_train else "测试"
    print_subsection(f"SeqData - {dataset_name} ({split_name})")
    
    try:
        # 加载数据集
        if split:
            seq_dataset = SeqData(
                root=dataset_folder,
                dataset=dataset_type,
                is_train=is_train,
                subsample=False,
                force_process=False,
                split=split
            )
        else:
            seq_dataset = SeqData(
                root=dataset_folder,
                dataset=dataset_type,
                is_train=is_train,
                subsample=False,
                force_process=False
            )
        
        print(f"数据集长度: {len(seq_dataset)}")
        print(f"最大序列长度: {seq_dataset.max_seq_len}")
        
        # 检查单个样本
        sample = seq_dataset[0]
        print(f"样本类型: {type(sample)}")
        
        stats = {}
        for field_name in sample._fields:
            field_value = getattr(sample, field_name)
            if isinstance(field_value, torch.Tensor):
                field_stats = examine_tensor_stats(field_value, field_name)
                stats[field_name] = field_stats
                
                print(f"  {field_name}:")
                print(f"    形状: {field_stats['shape']}")
                print(f"    数据类型: {field_stats['dtype']}")
                print(f"    内存: {field_stats['memory_mb']:.2f} MB")
                if 'min' in field_stats:
                    print(f"    值范围: [{field_stats['min']}, {field_stats['max']}]")
                if 'padding_ratio' in field_stats:
                    print(f"    填充比例: {field_stats['padding_ratio']:.2%}")
        
        # 检查多个样本的序列长度分布
        print(f"\n  序列长度统计 (检查前100个样本):")
        seq_lengths = []
        for i in range(min(100, len(seq_dataset))):
            sample = seq_dataset[i]
            if hasattr(sample, 'seq_mask'):
                seq_len = sample.seq_mask.sum().item()
                seq_lengths.append(seq_len)
        
        if seq_lengths:
            seq_lengths = np.array(seq_lengths)
            print(f"    平均长度: {seq_lengths.mean():.2f}")
            print(f"    长度范围: [{seq_lengths.min()}, {seq_lengths.max()}]")
            print(f"    标准差: {seq_lengths.std():.2f}")
        
        # 检查batch
        dataloader = DataLoader(seq_dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"\n  批次形状 (batch_size=4):")
        for field_name in batch._fields:
            field_value = getattr(batch, field_name)
            if isinstance(field_value, torch.Tensor):
                print(f"    {field_name}: {field_value.shape}")
        
        return {
            'dataset_length': len(seq_dataset),
            'max_seq_len': seq_dataset.max_seq_len,
            'sample_stats': stats,
            'seq_length_stats': {
                'mean': seq_lengths.mean() if len(seq_lengths) > 0 else 0,
                'min': seq_lengths.min() if len(seq_lengths) > 0 else 0,
                'max': seq_lengths.max() if len(seq_lengths) > 0 else 0,
                'std': seq_lengths.std() if len(seq_lengths) > 0 else 0,
            },
            'batch_shapes': {field: getattr(batch, field).shape for field in batch._fields 
                           if isinstance(getattr(batch, field), torch.Tensor)}
        }
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def examine_raw_data_files(dataset_name: str, dataset_type: RecDataset, 
                          dataset_folder: str, split: str = None) -> Dict[str, Any]:
    """检查原始数据文件"""
    print_subsection(f"原始数据文件 - {dataset_name}")
    
    try:
        # 加载原始数据集来检查处理后的HeteroData
        if dataset_type == RecDataset.AMAZON:
            from data.amazon import AmazonReviews
            if split:
                raw_dataset = AmazonReviews(root=dataset_folder, split=split)
            else:
                raw_dataset = AmazonReviews(root=dataset_folder, split="beauty")
        elif dataset_type == RecDataset.ML_32M:
            from data.ml32m import RawMovieLens32M
            raw_dataset = RawMovieLens32M(root=dataset_folder)
        else:  # ML_1M
            from data.ml1m import RawMovieLens1M
            raw_dataset = RawMovieLens1M(root=dataset_folder)
        
        data = raw_dataset.data
        
        print(f"HeteroData节点类型: {list(data.node_types)}")
        print(f"HeteroData边类型: {list(data.edge_types)}")
        
        # 检查item节点数据
        if 'item' in data:
            print(f"\n  item节点数据:")
            for key, value in data['item'].items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}, {value.dtype}")
                elif isinstance(value, np.ndarray):
                    print(f"    {key}: {value.shape}, {value.dtype}")
                else:
                    print(f"    {key}: {type(value)}")
        
        # 检查边数据
        edge_key = ("user", "rated", "item")
        if edge_key in data:
            print(f"\n  {edge_key}边数据:")
            edge_data = data[edge_key]
            for key, value in edge_data.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print(f"      {sub_key}: {sub_value.shape}, {sub_value.dtype}")
                        else:
                            print(f"      {sub_key}: {type(sub_value)}")
        
        return {'success': True}
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def examine_dataset(dataset_name: str, dataset_type: RecDataset, 
                   dataset_folder: str, split: str = None) -> Dict[str, Any]:
    """完整检查一个数据集"""
    print_separator(f"检查数据集: {dataset_name}")
    
    results = {}
    
    # 1. 检查原始数据文件
    results['raw_data'] = examine_raw_data_files(dataset_name, dataset_type, dataset_folder, split)
    
    # 2. 检查ItemData
    results['item_data'] = examine_item_dataset(dataset_name, dataset_type, dataset_folder, split)
    
    # 3. 检查SeqData (训练集)
    results['seq_data_train'] = examine_seq_dataset(dataset_name, dataset_type, dataset_folder, True, split)
    
    # 4. 检查SeqData (测试集)
    results['seq_data_test'] = examine_seq_dataset(dataset_name, dataset_type, dataset_folder, False, split)
    
    return results


def main():
    """主函数"""
    print_separator("数据集形状检查工具", "=", 80)
    
    # 数据集配置
    datasets_config = [
        {
            'name': 'Amazon Beauty',
            'type': RecDataset.AMAZON,
            'folder': 'dataset/amazon',
            'split': 'beauty'
        },
        {
            'name': 'MovieLens 32M',
            'type': RecDataset.ML_32M,
            'folder': 'dataset/ml-32m',
            'split': None
        },
        {
            'name': 'MovieLens 1M',
            'type': RecDataset.ML_1M,
            'folder': 'dataset/ml-1m',
            'split': None
        }
    ]
    
    all_results = {}
    
    for config in datasets_config:
        try:
            print(f"\n正在检查 {config['name']}...")
            results = examine_dataset(
                config['name'], 
                config['type'], 
                config['folder'], 
                config['split']
            )
            all_results[config['name']] = results
            
        except Exception as e:
            print(f"检查 {config['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results[config['name']] = {'error': str(e)}
    
    # 总结报告
    print_separator("总结报告")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        
        if 'error' in results:
            print(f"  ❌ 检查失败: {results['error']}")
            continue
        
        # ItemData总结
        if 'item_data' in results and 'error' not in results['item_data']:
            item_data = results['item_data']
            print(f"  📦 ItemData: {item_data['dataset_length']} 个物品")
        
        # SeqData总结
        if 'seq_data_train' in results and 'error' not in results['seq_data_train']:
            train_data = results['seq_data_train']
            print(f"  🚂 训练序列: {train_data['dataset_length']} 个")
            print(f"     最大序列长度: {train_data['max_seq_len']}")
            
        if 'seq_data_test' in results and 'error' not in results['seq_data_test']:
            test_data = results['seq_data_test']
            print(f"  🧪 测试序列: {test_data['dataset_length']} 个")
    
    print_separator("检查完成")


if __name__ == "__main__":
    main()
