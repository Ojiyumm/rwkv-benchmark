#!/usr/bin/env python3
"""
Evaluation Pipeline Runner
使用示例和快速启动脚本
"""

import os
import argparse
from pipeline import EvaluationPipeline, PipelineBuilder, quick_eval
from tasks import DatasetRegistry
from evaluators import EvaluatorRegistry


def example_1_single_task():
    """示例1: 运行单个评估任务（LAMBADA）"""
    print("\n" + "="*80)
    print("Example 1: Single Task Evaluation (LAMBADA)")
    print("="*80)
    
    model_path = "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"
    dataset_path = "/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl"
    
    # 方法1: 使用 quick_eval
    results = quick_eval(
        model_path=model_path,
        dataset_name="lambada",
        dataset_path=dataset_path,
        evaluator_name="perplexity",
        batch_size=256,
        limit=None  # 设置为 100 可以快速测试
    )
    
    print("\nResults:")
    for key, value in results['metrics'].items():
        print(f"  {key}: {value}")


def example_2_multiple_tasks():
    """示例2: 运行多个评估任务"""
    print("\n" + "="*80)
    print("Example 2: Multiple Tasks Evaluation")
    print("="*80)
    
    model_path = "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"
    
    # 使用 PipelineBuilder
    results = (PipelineBuilder(
        model_path=model_path,
        output_dir="./eval_results"
    )
    .add_task(
        dataset_name="lambada",
        dataset_path="/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl",
        evaluator_name="perplexity",
        batch_size=256,
        limit=100  # 快速测试
    )
    # 可以继续添加更多任务
    # .add_task(
    #     dataset_name="qa",
    #     dataset_path="/path/to/qa.jsonl",
    #     evaluator_name="generation",
    #     batch_size=128,
    #     max_length=100
    # )
    .run())
    
    print(f"\nCompleted {len(results)} tasks")


def example_3_custom_pipeline():
    """示例3: 使用 Pipeline 类（更灵活）"""
    print("\n" + "="*80)
    print("Example 3: Custom Pipeline")
    print("="*80)
    
    model_path = "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"
    
    # 创建管线
    pipeline = EvaluationPipeline(
        model_path=model_path,
        output_dir="./eval_results",
        seed=42
    )
    
    # 任务1: LAMBADA (困惑度)
    results1 = pipeline.run(
        dataset_name="lambada",
        dataset_path="/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl",
        evaluator_name="perplexity",
        batch_size=256,
        limit=100,
        print_interval=50
    )
    
    print(f"\nLAMBADA Perplexity: {results1['metrics']['perplexity']:.2f}")
    print(f"LAMBADA Accuracy: {results1['metrics']['accuracy']*100:.1f}%")


def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(
        description="RWKV Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行 LAMBADA 评估
  python run_pipeline.py --dataset lambada \\
      --dataset-path /path/to/lambada.jsonl \\
      --evaluator perplexity \\
      --batch-size 256
  
  # 快速测试（限制100个样本）
  python run_pipeline.py --dataset lambada \\
      --dataset-path /path/to/lambada.jsonl \\
      --evaluator perplexity \\
      --limit 100
  
  # 运行示例
  python run_pipeline.py --example 1
  
  # 列出可用的数据集和评估器
  python run_pipeline.py --list
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_PATH", "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"),
        help="Path to RWKV model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Output directory for results"
    )
    
    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (registered in DatasetRegistry)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset file"
    )
    
    # 评估器参数
    parser.add_argument(
        "--evaluator",
        type=str,
        help="Evaluator name (registered in EvaluatorRegistry)"
    )
    
    # 可选参数
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (default: use dataset default)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Max generation length (default: use dataset default)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples (for quick testing)"
    )
    parser.add_argument(
        "--inferoutput",
        type=str,
        help="Path to save inference results (questions and answers) in JSONL format"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # 特殊模式
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3],
        help="Run example (1, 2, or 3)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and evaluators"
    )
    
    args = parser.parse_args()
    
    # 列出可用组件
    if args.list:
        print("\n" + "="*80)
        print("Available Components")
        print("="*80)
        
        print("\nDatasets:")
        for name in DatasetRegistry.list_datasets():
            config = DatasetRegistry.get(name)
            print(f"  • {name:15s} - {config.description}")
            print(f"    {'':15s}   (batch_size={config.default_batch_size}, max_length={config.default_max_length})")
        
        print("\nEvaluators:")
        for name in EvaluatorRegistry.list_evaluators():
            config = EvaluatorRegistry.get(name)
            print(f"  • {name:15s} - {config.description}")
            print(f"    {'':15s}   metrics: {', '.join(config.metrics)}")
        
        print()
        return
    
    # 运行示例
    if args.example:
        if args.example == 1:
            example_1_single_task()
        elif args.example == 2:
            example_2_multiple_tasks()
        elif args.example == 3:
            example_3_custom_pipeline()
        return
    
    # 正常评估模式
    if not args.dataset or not args.dataset_path or not args.evaluator:
        parser.print_help()
        print("\n❌ Error: --dataset, --dataset-path, and --evaluator are required")
        print("   Or use --example or --list\n")
        return
    
    # 运行评估
    print("\n" + "="*80)
    print("Running Custom Evaluation")
    print("="*80)
    
    pipeline = EvaluationPipeline(
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # 构建评估参数
    eval_kwargs = {}
    if args.inferoutput:
        eval_kwargs['inferoutput'] = args.inferoutput
    
    results = pipeline.run(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        evaluator_name=args.evaluator,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
        **eval_kwargs
    )
    
    print("\n📊 Final Results:")
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    main()

