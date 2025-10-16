#!/usr/bin/env python3
"""
使用示例 - 展示如何使用新的模块化结构
"""

# 导入会自动注册所有数据集和评估器
from tasks import DatasetRegistry
from evaluators import EvaluatorRegistry
from pipeline import quick_eval, PipelineBuilder

def example_1_list_components():
    """示例1: 列出所有可用的组件"""
    print("\n" + "="*80)
    print("Example 1: 列出所有可用组件")
    print("="*80)
    
    print("\n可用数据集:")
    for name in DatasetRegistry.list_datasets():
        config = DatasetRegistry.get(name)
        print(f"  • {name:15s} - {config.description}")
    
    print("\n可用评估器:")
    for name in EvaluatorRegistry.list_evaluators():
        config = EvaluatorRegistry.get(name)
        print(f"  • {name:15s} - {config.description}")


def example_2_quick_eval():
    """示例2: 快速评估（单个任务）"""
    print("\n" + "="*80)
    print("Example 2: 快速评估")
    print("="*80)
    
    model_path = "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"
    dataset_path = "/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl"
    
    results = quick_eval(
        model_path=model_path,
        dataset_name="lambada",
        dataset_path=dataset_path,
        evaluator_name="perplexity",
        batch_size=256,
        limit=100  # 只测试100个样本
    )
    
    print("\n结果:")
    print(f"  Perplexity: {results['metrics']['perplexity']:.2f}")
    print(f"  Accuracy: {results['metrics']['accuracy']*100:.1f}%")


def example_3_pipeline_builder():
    """示例3: 使用 PipelineBuilder 运行多个任务"""
    print("\n" + "="*80)
    print("Example 3: 批量运行多个任务")
    print("="*80)
    
    model_path = "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"
    
    results = (PipelineBuilder(
        model_path=model_path,
        output_dir="./example_results"
    )
    .add_task(
        dataset_name="lambada",
        dataset_path="/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl",
        evaluator_name="perplexity",
        batch_size=256,
        limit=50  # 快速测试
    )
    .add_task(
        dataset_name="mmlu_pro",
        dataset_path="/path/to/mmlu_pro.jsonl",  # 修改为实际路径
        evaluator_name="mmlu_pro",
        batch_size=64,
        limit=20  # 快速测试
    )
    .run())
    
    print(f"\n完成 {len(results)} 个任务")
    for i, result in enumerate(results, 1):
        print(f"\n任务 {i}: {result['dataset']['name']}")
        print(f"  主要指标: {list(result['metrics'].keys())}")


def example_4_custom_dataset():
    """示例4: 临时注册自定义数据集"""
    print("\n" + "="*80)
    print("Example 4: 注册自定义数据集")
    print("="*80)
    
    import json
    
    # 定义自定义加载器和prompt模板
    def load_my_data(path: str):
        with open(path, 'r') as f:
            return json.load(f)
    
    def my_prompt(item: dict) -> str:
        return f"输入: {item['input']}\n输出:"
    
    # 注册
    DatasetRegistry.register(
        name='my_custom_dataset',
        loader=load_my_data,
        prompt_template=my_prompt,
        description='我的自定义数据集',
        default_batch_size=64,
        default_max_length=100
    )
    
    print("\n✓ 自定义数据集已注册")
    print(f"\n现在可用的数据集: {DatasetRegistry.list_datasets()}")
    
    # 使用自定义数据集
    # results = quick_eval(
    #     model_path="/path/to/model",
    #     dataset_name="my_custom_dataset",
    #     dataset_path="/path/to/my_data.json",
    #     evaluator_name="generation"
    # )


def main():
    """运行所有示例"""
    import sys
    
    print("\n" + "="*80)
    print("RWKV 评估框架 - 使用示例")
    print("="*80)
    
    # 示例1: 列出组件
    example_1_list_components()
    
    # 如果提供了 --demo 参数，运行实际评估
    if "--demo" in sys.argv:
        print("\n" + "="*80)
        print("运行实际评估示例...")
        print("="*80)
        
        # 示例2: 快速评估
        example_2_quick_eval()
        
        # 示例3: 批量评估（需要实际数据路径）
        # example_3_pipeline_builder()
    
    # 示例4: 自定义数据集（演示注册过程）
    example_4_custom_dataset()
    
    print("\n" + "="*80)
    print("使用说明:")
    print("="*80)
    print("\n1. 列出组件:")
    print("   python example_usage.py")
    print("\n2. 运行实际评估:")
    print("   python example_usage.py --demo")
    print("\n3. 查看完整文档:")
    print("   cat README.md")
    print()


if __name__ == "__main__":
    main()

