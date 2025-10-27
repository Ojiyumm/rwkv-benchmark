#!/usr/bin/env python3
"""
运行 MMLU Pro 评估
MMLU Pro 会在导入 datasets 模块时自动注册
"""

import sys
import argparse
from pipeline import quick_eval, PipelineBuilder

# tasks 模块会自动注册所有数据集和评估器
import tasks
import evaluators


def main():
    parser = argparse.ArgumentParser(description='Run MMLU Pro Evaluation')
    
    parser.add_argument(
        '--model-path',
        type=str,
        default="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096",
        help='Path to RWKV model'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to MMLU Pro dataset file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=10,
        help='Max generation length (use 200 for CoT)'
    )
    parser.add_argument(
        '--cot',
        action='store_true',
        help='Use Chain-of-Thought prompting'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of samples for quick testing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./eval_results',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # 选择数据集版本
    dataset_name = 'mmlu_pro_cot' if args.cot else 'mmlu_pro'
    
    print("\n" + "="*80)
    print(f"MMLU Pro Evaluation {'(Chain-of-Thought)' if args.cot else ''}")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    if args.limit:
        print(f"Limit: {args.limit} samples")
    print("="*80 + "\n")
    
    # 运行评估
    results = quick_eval(
        model_path=args.model_path,
        dataset_name=dataset_name,
        dataset_path=args.dataset_path,
        evaluator_name='mmlu_pro',
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
        use_cot=args.cot
    )
    
    # 打印结果摘要
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    metrics = results['metrics']
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    print(f"Subjects: {metrics['num_subjects']}")
    
    if 'subject_accuracies' in metrics:
        print("\nTop 5 Best Subjects:")
        sorted_subjects = sorted(
            metrics['subject_accuracies'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for subject, acc in sorted_subjects[:5]:
            print(f"  {subject:30s}: {acc*100:.2f}%")
        
        print("\nTop 5 Worst Subjects:")
        for subject, acc in sorted_subjects[-5:]:
            print(f"  {subject:30s}: {acc*100:.2f}%")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

