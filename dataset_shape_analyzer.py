import torch
import numpy as np
from torch_geometric.data import HeteroData
import os
from typing import Dict, Any, List, Tuple
import pandas as pd

class DatasetShapeAnalyzer:
    """数据集结构和shape分析器"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_tensor_shape(self, data: Any, name: str = "", max_depth: int = 3, current_depth: int = 0) -> Dict:
        """递归分析数据结构和shape"""
        result = {
            'name': name,
            'type': str(type(data).__name__),
            'shape': None,
            'dtype': None,
            'size': None,
            'children': {}
        }
        
        if current_depth >= max_depth:
            result['note'] = 'Max depth reached'
            return result
        
        try:
            # Tensor类型
            if torch.is_tensor(data):
                result['shape'] = list(data.shape)
                result['dtype'] = str(data.dtype)
                result['size'] = data.numel()
                result['device'] = str(data.device)
                
                # 如果是小张量，显示部分数据
                if data.numel() <= 20:
                    result['sample_data'] = data.tolist()
                else:
                    result['sample_data'] = f"First 5 elements: {data.flatten()[:5].tolist()}"
            
            # NumPy数组
            elif isinstance(data, np.ndarray):
                result['shape'] = list(data.shape)
                result['dtype'] = str(data.dtype)
                result['size'] = data.size
                if data.size <= 20:
                    result['sample_data'] = data.tolist()
                else:
                    result['sample_data'] = f"First 5 elements: {data.flatten()[:5].tolist()}"
            
            # 字典类型
            elif isinstance(data, dict):
                result['length'] = len(data)
                result['keys'] = list(data.keys())
                for key, value in data.items():
                    if current_depth < max_depth - 1:
                        result['children'][str(key)] = self.analyze_tensor_shape(
                            value, str(key), max_depth, current_depth + 1
                        )
            
            # 列表或元组
            elif isinstance(data, (list, tuple)):
                result['length'] = len(data)
                if len(data) > 0:
                    # 分析前几个元素的类型
                    first_elem = data[0]
                    result['element_type'] = str(type(first_elem).__name__)
                    
                    if len(data) <= 10:  # 如果列表较短，分析所有元素
                        for i, item in enumerate(data[:5]):  # 只分析前5个
                            result['children'][f'[{i}]'] = self.analyze_tensor_shape(
                                item, f'[{i}]', max_depth, current_depth + 1
                            )
                    else:
                        # 分析第一个元素
                        result['children']['[0]'] = self.analyze_tensor_shape(
                            first_elem, '[0]', max_depth, current_depth + 1
                        )
            
            # HeteroData特殊处理
            elif isinstance(data, HeteroData):
                result['node_types'] = list(data.node_types)
                result['edge_types'] = [str(edge_type) for edge_type in data.edge_types]
                result['metadata'] = str(data.metadata())
                
                # 分析节点数据
                for node_type in data.node_types:
                    node_data = data[node_type]
                    result['children'][f'node_{node_type}'] = self.analyze_tensor_shape(
                        node_data, f'node_{node_type}', max_depth, current_depth + 1
                    )
                
                # 分析边数据
                for edge_type in data.edge_types:
                    edge_data = data[edge_type]
                    edge_key = f"edge_{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
                    result['children'][edge_key] = self.analyze_tensor_shape(
                        edge_data, edge_key, max_depth, current_depth + 1
                    )
            
            # 其他对象
            else:
                if hasattr(data, '__len__'):
                    result['length'] = len(data)
                
                # 尝试获取常见属性
                common_attrs = ['x', 'y', 'edge_index', 'edge_attr', 'batch', 'ptr']
                for attr in common_attrs:
                    if hasattr(data, attr):
                        attr_data = getattr(data, attr)
                        if attr_data is not None:
                            result['children'][attr] = self.analyze_tensor_shape(
                                attr_data, attr, max_depth, current_depth + 1
                            )
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def print_analysis(self, analysis: Dict, indent: str = "") -> None:
        """打印分析结果"""
        name = analysis['name']
        type_name = analysis['type']
        
        print(f"{indent}📊 {name} ({type_name})")
        
        if analysis.get('shape'):
            print(f"{indent}  Shape: {analysis['shape']}")
        if analysis.get('dtype'):
            print(f"{indent}  Dtype: {analysis['dtype']}")
        if analysis.get('size'):
            print(f"{indent}  Size: {analysis['size']:,}")
        if analysis.get('device'):
            print(f"{indent}  Device: {analysis['device']}")
        if analysis.get('length'):
            print(f"{indent}  Length: {analysis['length']}")
        if analysis.get('keys'):
            print(f"{indent}  Keys: {analysis['keys']}")
        if analysis.get('node_types'):
            print(f"{indent}  Node Types: {analysis['node_types']}")
        if analysis.get('edge_types'):
            print(f"{indent}  Edge Types: {analysis['edge_types']}")
        if analysis.get('sample_data'):
            print(f"{indent}  Sample: {analysis['sample_data']}")
        if analysis.get('error'):
            print(f"{indent}  ❌ Error: {analysis['error']}")
        
        # 递归打印子结构
        if analysis.get('children'):
            for child_name, child_analysis in analysis['children'].items():
                print()
                self.print_analysis(child_analysis, indent + "  ")
    
    def compare_datasets(self, data1: Any, data2: Any, name1: str, name2: str) -> None:
        """比较两个数据集的结构"""
        print("="*80)
        print(f"🔍 数据集结构对比: {name1} vs {name2}")
        print("="*80)
        
        analysis1 = self.analyze_tensor_shape(data1, name1)
        analysis2 = self.analyze_tensor_shape(data2, name2)
        
        print(f"\n📈 {name1} 数据结构:")
        print("-" * 50)
        self.print_analysis(analysis1)
        
        print(f"\n📈 {name2} 数据结构:")
        print("-" * 50)
        self.print_analysis(analysis2)
        
        # 结构差异分析
        print(f"\n🔄 结构差异分析:")
        print("-" * 50)
        self._compare_structures(analysis1, analysis2, name1, name2)
    
    def _compare_structures(self, analysis1: Dict, analysis2: Dict, name1: str, name2: str) -> None:
        """比较两个分析结果的差异"""
        print(f"类型: {name1}({analysis1['type']}) vs {name2}({analysis2['type']})")
        
        if analysis1.get('shape') and analysis2.get('shape'):
            print(f"Shape: {name1}{analysis1['shape']} vs {name2}{analysis2['shape']}")
        
        if analysis1.get('keys') and analysis2.get('keys'):
            keys1 = set(analysis1['keys'])
            keys2 = set(analysis2['keys'])
            common = keys1 & keys2
            only1 = keys1 - keys2
            only2 = keys2 - keys1
            
            print(f"共同keys ({len(common)}): {sorted(common)}")
            if only1:
                print(f"仅{name1}有 ({len(only1)}): {sorted(only1)}")
            if only2:
                print(f"仅{name2}有 ({len(only2)}): {sorted(only2)}")

def safe_torch_load(data_path: str) -> Any:
    """安全加载torch数据文件，兼容PyTorch 2.6+"""
    try:
        # 尝试使用新的安全加载方式
        from numpy.core.multiarray import _reconstruct
        import torch.serialization
        
        # 方法1: 使用安全的全局变量上下文管理器
        with torch.serialization.safe_globals([_reconstruct]):
            raw_data = torch.load(data_path, weights_only=True)
        data = extract_actual_data(raw_data)
        return data
    except:
        try:
            # 方法2: 如果上面失败，使用传统方式（需要信任数据源）
            print("⚠️  使用传统加载方式 (weights_only=False)")
            raw_data = torch.load(data_path, weights_only=False)
            data = extract_actual_data(raw_data)
            return data
        except Exception as e:
            print(f"❌ 所有加载方式都失败: {e}")
            return None

def extract_actual_data(raw_data: Any) -> Any:
    """从PyTorch Geometric保存的数据中提取实际的数据对象"""
    # PyTorch Geometric的InMemoryDataset保存格式通常是tuple: (data_list, slices, data_cls)
    if isinstance(raw_data, tuple):
        print(f"🔍 检测到tuple格式数据，长度: {len(raw_data)}")
        
        # 通常第一个元素是数据列表或实际数据
        if len(raw_data) > 0:
            first_elem = raw_data[0]
            
            # 如果第一个元素是列表且包含数据
            if isinstance(first_elem, list) and len(first_elem) > 0:
                print(f"✅ 提取列表中的第一个数据对象")
                return first_elem[0]
            
            # 如果第一个元素是字典（可能包含HeteroData的存储）
            elif isinstance(first_elem, dict):
                print(f"✅ 检测到字典格式，尝试重构HeteroData")
                return reconstruct_hetero_data_from_dict(first_elem)
            
            # 如果第一个元素直接是数据对象
            elif first_elem is not None and not isinstance(first_elem, type):
                print(f"✅ 提取第一个元素作为数据对象")
                return first_elem
            
            # 如果第一个元素是None，尝试其他元素
            else:
                for i, elem in enumerate(raw_data):
                    if elem is not None and not isinstance(elem, type):
                        if isinstance(elem, dict):
                            print(f"✅ 从第{i}个元素重构HeteroData")
                            return reconstruct_hetero_data_from_dict(elem)
                        elif hasattr(elem, '__dict__'):
                            print(f"✅ 提取第{i}个元素作为数据对象")
                            return elem
    
    # 如果不是tuple或无法提取，返回原始数据
    print(f"🔄 返回原始数据格式: {type(raw_data)}")
    return raw_data

def reconstruct_hetero_data_from_dict(data_dict: dict) -> HeteroData:
    """从字典重构HeteroData对象"""
    try:
        hetero_data = HeteroData()
        
        # 处理节点数据
        for key, value in data_dict.items():
            if key == '_global_store':
                continue
            elif isinstance(key, str) and key in ['item', 'user']:
                # 节点数据
                node_type = key
                if isinstance(value, dict):
                    for attr_name, attr_value in value.items():
                        setattr(hetero_data[node_type], attr_name, attr_value)
                        print(f"  添加节点 {node_type}.{attr_name}: {type(attr_value)}")
            elif isinstance(key, tuple) and len(key) == 3:
                # 边数据
                edge_type = key
                if isinstance(value, dict):
                    for attr_name, attr_value in value.items():
                        setattr(hetero_data[edge_type], attr_name, attr_value)
                        print(f"  添加边 {edge_type}.{attr_name}: {type(attr_value)}")
        
        print(f"✅ 成功重构HeteroData")
        return hetero_data
        
    except Exception as e:
        print(f"❌ HeteroData重构失败: {e}")
        return data_dict

def analyze_ml32m_data(data_path: str) -> Any:
    """加载和分析ML32M数据"""
    print("🎬 加载MovieLens32M数据...")
    try:
        if os.path.exists(data_path):
            data = safe_torch_load(data_path)
            if data is not None:
                print(f"✅ 成功加载ML32M数据从: {data_path}")
                return data
            else:
                print(f"❌ 加载失败: {data_path}")
                return None
        else:
            print(f"❌ 文件不存在: {data_path}")
            return None
    except Exception as e:
        print(f"❌ 加载ML32M数据失败: {e}")
        return None

def analyze_amazon_data(data_path: str) -> Any:
    """加载和分析Amazon数据"""
    print("🛒 加载Amazon数据...")
    try:
        if os.path.exists(data_path):
            data = safe_torch_load(data_path)
            if data is not None:
                print(f"✅ 成功加载Amazon数据从: {data_path}")
                return data
            else:
                print(f"❌ 加载失败: {data_path}")
                return None
        else:
            print(f"❌ 文件不存在: {data_path}")
            return None
    except Exception as e:
        print(f"❌ 加载Amazon数据失败: {e}")
        return None

def detailed_hetero_analysis(data: HeteroData, dataset_name: str) -> None:
    """详细分析HeteroData结构"""
    print(f"\n🔬 {dataset_name} HeteroData详细分析:")
    print("=" * 60)
    
    # 基本信息
    print(f"📊 基本信息:")
    print(f"  节点类型: {data.node_types if hasattr(data, 'node_types') else '未知'}")
    print(f"  边类型: {[str(et) for et in data.edge_types] if hasattr(data, 'edge_types') else '未知'}")
    
    # 节点分析
    print("\n📦 节点类型分析:")
    if hasattr(data, 'node_types'):
        for node_type in data.node_types:
            node_data = data[node_type]
            print(f"  {node_type}:")
            for attr_name in dir(node_data):
                if not attr_name.startswith('_') and attr_name not in ['keys', 'items', 'values']:
                    try:
                        attr_value = getattr(node_data, attr_name)
                        if torch.is_tensor(attr_value):
                            print(f"    {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                        elif isinstance(attr_value, np.ndarray):
                            print(f"    {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                        elif attr_value is not None and not callable(attr_value):
                            if hasattr(attr_value, '__len__'):
                                print(f"    {attr_name}: {type(attr_value)} (len={len(attr_value)})")
                            else:
                                print(f"    {attr_name}: {type(attr_value)}")
                    except:
                        continue
    else:
        # 如果没有标准的node_types，尝试检查常见的节点类型
        common_nodes = ['user', 'item']
        for node_type in common_nodes:
            if hasattr(data, '__getitem__'):
                try:
                    node_data = data[node_type]
                    print(f"  {node_type}: 找到节点数据")
                    if hasattr(node_data, '__dict__'):
                        for attr_name, attr_value in node_data.__dict__.items():
                            if torch.is_tensor(attr_value):
                                print(f"    {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                            elif isinstance(attr_value, np.ndarray):
                                print(f"    {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                            elif attr_value is not None:
                                if hasattr(attr_value, '__len__'):
                                    print(f"    {attr_name}: {type(attr_value)} (len={len(attr_value)})")
                                else:
                                    print(f"    {attr_name}: {type(attr_value)}")
                except:
                    continue
    
    # 边分析
    print("\n🔗 边类型分析:")
    if hasattr(data, 'edge_types'):
        for edge_type in data.edge_types:
            edge_data = data[edge_type]
            print(f"  {edge_type}:")
            for attr_name in dir(edge_data):
                if not attr_name.startswith('_') and attr_name not in ['keys', 'items', 'values']:
                    try:
                        attr_value = getattr(edge_data, attr_name)
                        if torch.is_tensor(attr_value):
                            print(f"    {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                        elif isinstance(attr_value, np.ndarray):
                            print(f"    {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                        elif isinstance(attr_value, dict):
                            print(f"    {attr_name}: dict with keys {list(attr_value.keys())}")
                            # 特别处理history数据
                            if attr_name == 'history':
                                analyze_history_data(attr_value, dataset_name)
                        elif attr_value is not None and not callable(attr_value):
                            if hasattr(attr_value, '__len__'):
                                print(f"    {attr_name}: {type(attr_value)} (len={len(attr_value)})")
                            else:
                                print(f"    {attr_name}: {type(attr_value)}")
                    except:
                        continue
    else:
        # 尝试检查常见的边类型
        common_edges = [('user', 'rated', 'item'), ('user', 'rates', 'item')]
        for edge_type in common_edges:
            try:
                edge_data = data[edge_type]
                print(f"  {edge_type}: 找到边数据")
                if hasattr(edge_data, '__dict__'):
                    for attr_name, attr_value in edge_data.__dict__.items():
                        if torch.is_tensor(attr_value):
                            print(f"    {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                        elif isinstance(attr_value, dict):
                            print(f"    {attr_name}: dict with keys {list(attr_value.keys())}")
                            if attr_name == 'history':
                                analyze_history_data(attr_value, dataset_name)
                        elif attr_value is not None:
                            if hasattr(attr_value, '__len__'):
                                print(f"    {attr_name}: {type(attr_value)} (len={len(attr_value)})")
                            else:
                                print(f"    {attr_name}: {type(attr_value)}")
            except:
                continue

def analyze_history_data(history_data: dict, dataset_name: str) -> None:
    """分析history数据的详细结构"""
    print(f"\n    📚 {dataset_name} History数据详细分析:")
    for split_name, split_data in history_data.items():
        print(f"      {split_name}:")
        if hasattr(split_data, 'items'):
            for key, value in split_data.items():
                if torch.is_tensor(value):
                    print(f"        {key}: {list(value.shape)} ({value.dtype})")
                    if value.numel() <= 10:
                        print(f"          示例数据: {value.tolist()}")
                elif hasattr(value, '__len__'):
                    print(f"        {key}: {type(value)} (len={len(value)})")
                else:
                    print(f"        {key}: {type(value)}")
        else:
            print(f"        类型: {type(split_data)}")
            if hasattr(split_data, 'shape'):
                print(f"        形状: {split_data.shape}")

def main():
    """主函数：执行完整的数据分析"""
    analyzer = DatasetShapeAnalyzer()
    
    # 数据路径配置 - 请根据实际路径修改
    ml32m_path = "dataset/ml-32m/processed/data.pt"  # ML32M处理后的数据路径
    amazon_path = "dataset/amazon/processed/data_beauty.pt"  # Amazon处理后的数据路径
    
    print("🚀 开始数据集Shape分析")
    print("=" * 80)
    
    # 加载数据
    ml32m_data = analyze_ml32m_data(ml32m_path)
    amazon_data = analyze_amazon_data(amazon_path)
    
    # 如果两个数据集都成功加载，进行对比分析
    if ml32m_data is not None and amazon_data is not None:
        # 简化的快速对比
        print(f"\n🎯 快速数据对比:")
        print("=" * 50)
        quick_data_summary(ml32m_data, "ML32M")
        quick_data_summary(amazon_data, "Amazon")
        
        # 详细对比
        analyzer.compare_datasets(ml32m_data, amazon_data, "ML32M", "Amazon")
        
        # 详细的HeteroData分析
        if isinstance(ml32m_data, HeteroData):
            detailed_hetero_analysis(ml32m_data, "ML32M")
        
        if isinstance(amazon_data, HeteroData):
            detailed_hetero_analysis(amazon_data, "Amazon")
    
    # 单独分析可用的数据集
    elif ml32m_data is not None:
        print("\n📊 ML32M单独分析:")
        quick_data_summary(ml32m_data, "ML32M")
        analysis = analyzer.analyze_tensor_shape(ml32m_data, "ML32M")
        analyzer.print_analysis(analysis)
        if isinstance(ml32m_data, HeteroData):
            detailed_hetero_analysis(ml32m_data, "ML32M")
    
    elif amazon_data is not None:
        print("\n📊 Amazon单独分析:")
        quick_data_summary(amazon_data, "Amazon")
        analysis = analyzer.analyze_tensor_shape(amazon_data, "Amazon")
        analyzer.print_analysis(analysis)
        if isinstance(amazon_data, HeteroData):
            detailed_hetero_analysis(amazon_data, "Amazon")
    
    else:
        print("❌ 无法加载任何数据集，请检查数据路径")
        print("💡 提示：请修改main()函数中的ml32m_path和amazon_path变量")

def quick_data_summary(data: Any, dataset_name: str) -> None:
    """快速数据摘要，显示关键信息"""
    print(f"\n📋 {dataset_name} 数据摘要:")
    print("-" * 30)
    
    if isinstance(data, HeteroData):
        # HeteroData格式
        print(f"  格式: HeteroData")
        if hasattr(data, 'node_types'):
            print(f"  节点类型: {data.node_types}")
        if hasattr(data, 'edge_types'):
            print(f"  边类型: {len(data.edge_types)} 种")
        
        # 检查关键数据
        try:
            if 'item' in str(data.node_types):
                item_data = data['item']
                if hasattr(item_data, 'x') and torch.is_tensor(item_data.x):
                    print(f"  物品特征: {list(item_data.x.shape)}")
                if hasattr(item_data, 'text'):
                    print(f"  物品文本: {len(item_data.text) if hasattr(item_data.text, '__len__') else 'N/A'}")
            
            # 检查边数据
            edge_keys = [('user', 'rated', 'item'), ('user', 'rates', 'item')]
            for edge_key in edge_keys:
                try:
                    edge_data = data[edge_key]
                    if hasattr(edge_data, 'history'):
                        history = edge_data.history
                        print(f"  历史数据splits: {list(history.keys())}")
                        if 'train' in history:
                            train_data = history['train']
                            if hasattr(train_data, 'items'):
                                print(f"  训练数据字段: {list(train_data.keys())}")
                    break
                except:
                    continue
                    
        except Exception as e:
            print(f"  ⚠️  数据检查出错: {e}")
            
    else:
        print(f"  格式: {type(data)}")
        if hasattr(data, '__len__'):
            print(f"  长度: {len(data)}")
        if hasattr(data, 'shape'):
            print(f"  形状: {data.shape}")

# 快速测试函数
def quick_shape_check(data_path: str, dataset_name: str):
    """快速检查单个数据集的shape"""
    print(f"🔍 快速检查 {dataset_name} 数据shape:")
    print("-" * 40)
    
    try:
        data = safe_torch_load(data_path)
        if data is None:
            return
            
        analyzer = DatasetShapeAnalyzer()
        analysis = analyzer.analyze_tensor_shape(data, dataset_name, max_depth=2)
        analyzer.print_analysis(analysis)
        
        if isinstance(data, HeteroData):
            print(f"\n📋 {dataset_name} 基本信息:")
            print(f"  节点类型: {data.node_types}")
            print(f"  边类型: {[str(et) for et in data.edge_types]}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
    
    # 如果需要快速检查单个数据集，取消下面的注释
    # quick_shape_check("path/to/your/ml32m/data.pt", "ML32M")
    # quick_shape_check("path/to/your/amazon/data.pt", "Amazon")