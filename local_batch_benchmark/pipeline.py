"""
Evaluation Pipeline - 评估管线
串联数据集加载、批量推理和评估
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from batch_engine import RWKVInferenceEngine
from tasks import DatasetRegistry
from evaluators import EvaluatorRegistry


class EvaluationPipeline:
    """评估管线"""
    
    def __init__(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        seed: int = 42,
        output_dir: str = "./eval_results"
    ):
        """
        初始化评估管线
        
        Args:
            model_path: 模型路径
            vocab_path: 词汇表路径
            seed: 随机种子
            output_dir: 输出目录
        """
        print("\n" + "="*80)
        print("Initializing Evaluation Pipeline")
        print("="*80)
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化推理引擎
        print(f"\nModel: {model_path}")
        self.engine = RWKVInferenceEngine(
            model_path=model_path,
            vocab_path=vocab_path,
            seed=seed
        )
        
        print("✓ Pipeline initialized\n")
    
    def run(
        self,
        dataset_name: str,
        dataset_path: str,
        evaluator_name: str,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        limit: Optional[int] = None,
        save_results: bool = True,
        **eval_kwargs
    ) -> Dict[str, Any]:
        """
        运行完整的评估管线
        
        Args:
            dataset_name: 数据集名称（已注册）
            dataset_path: 数据集文件路径
            evaluator_name: 评估器名称（已注册）
            batch_size: 批量大小（None则使用数据集默认值）
            max_length: 最大生成长度（None则使用数据集默认值）
            limit: 限制样本数量（用于快速测试）
            save_results: 是否保存结果
            **eval_kwargs: 传递给评估器的额外参数
            
        Returns:
            评估结果字典
        """
        print("\n" + "="*80)
        print(f"Running Evaluation: {dataset_name} + {evaluator_name}")
        print("="*80)
        
        # 1. 加载数据集
        print(f"\n[1/3] Loading dataset: {dataset_name}")
        print(f"      Path: {dataset_path}")
        
        data = DatasetRegistry.load_dataset(
            name=dataset_name,
            path=dataset_path,
            limit=limit
        )
        
        # 获取数据集默认配置
        dataset_config = DatasetRegistry.get(dataset_name)
        if batch_size is None:
            batch_size = dataset_config.default_batch_size
        if max_length is None:
            max_length = dataset_config.default_max_length
        
        print(f"      Batch size: {batch_size}")
        print(f"      Max length: {max_length}")
        
        # 2. 运行评估
        print(f"\n[2/3] Running evaluator: {evaluator_name}")
        
        eval_results = EvaluatorRegistry.evaluate(
            name=evaluator_name,
            engine=self.engine,
            data=data,
            batch_size=batch_size,
            max_length=max_length,
            **eval_kwargs
        )
        
        # 附加速度测量（与benchmark一致的batch解码环）
        try:
            speed = self.engine.measure_decode_speed(batch_size=batch_size, length=32)
            print(f"\n[Speed] BSZ {int(speed['batch_size'])} || Token/s = {speed['forward_tps']:.2f} (forward), {speed['full_tps']:.2f} (full)")
            # 合并进metrics，便于保存
            eval_results = {
                **eval_results,
                'forward_tps': round(speed['forward_tps'], 2),
                'full_tps': round(speed['full_tps'], 2),
                'step_forward_ms_p50': round(speed['step_forward_ms_p50'], 3),
                'step_full_ms_p50': round(speed['step_full_ms_p50'], 3)
            }
        except Exception as e:
            print(f"[Speed] Measure failed: {e}")
        
        # 3. 保存结果
        print(f"\n[3/3] Finalizing results")
        
        # 构建完整结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_results = {
            'timestamp': timestamp,
            'dataset': {
                'name': dataset_name,
                'path': dataset_path,
                'size': len(data),
                'limit': limit
            },
            'evaluator': evaluator_name,
            'config': {
                'batch_size': batch_size,
                'max_length': max_length,
                **eval_kwargs
            },
            'metrics': eval_results
        }
        
        if save_results:
            # 保存 JSON 结果
            result_file = os.path.join(
                self.output_dir,
                f"{dataset_name}_{evaluator_name}_{timestamp}.json"
            )
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results saved to: {result_file}")
        
        print("\n" + "="*80)
        print("Evaluation completed!")
        print("="*80 + "\n")
        
        return full_results
    
    def run_multiple(
        self,
        tasks: List[Dict[str, Any]],
        save_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        运行多个评估任务
        
        Args:
            tasks: 任务列表，每个任务是一个字典，包含 run() 方法的参数
            save_summary: 是否保存汇总结果
            
        Returns:
            所有评估结果的列表
        """
        print("\n" + "="*80)
        print(f"Running Multiple Evaluations ({len(tasks)} tasks)")
        print("="*80 + "\n")
        
        all_results = []
        
        for i, task in enumerate(tasks, 1):
            print(f"\n>>> Task {i}/{len(tasks)}")
            result = self.run(**task)
            all_results.append(result)
        
        if save_summary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(
                self.output_dir,
                f"summary_{timestamp}.json"
            )
            
            summary = {
                'timestamp': timestamp,
                'total_tasks': len(tasks),
                'results': all_results
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Summary saved to: {summary_file}")
        
        print("\n" + "="*80)
        print(f"All {len(tasks)} evaluations completed!")
        print("="*80 + "\n")
        
        return all_results


class PipelineBuilder:
    """管线构建器 - 便捷接口"""
    
    def __init__(self, model_path: str, output_dir: str = "./eval_results"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.tasks = []
    
    def add_task(
        self,
        dataset_name: str,
        dataset_path: str,
        evaluator_name: str,
        **kwargs
    ):
        """添加评估任务"""
        self.tasks.append({
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'evaluator_name': evaluator_name,
            **kwargs
        })
        return self
    
    def run(self) -> List[Dict[str, Any]]:
        """运行所有任务"""
        pipeline = EvaluationPipeline(
            model_path=self.model_path,
            output_dir=self.output_dir
        )
        return pipeline.run_multiple(self.tasks)


# ==================== 便捷函数 ====================

def quick_eval(
    model_path: str,
    dataset_name: str,
    dataset_path: str,
    evaluator_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    快速评估（单个任务）
    
    Example:
        results = quick_eval(
            model_path="/path/to/model",
            dataset_name="lambada",
            dataset_path="/path/to/lambada.jsonl",
            evaluator_name="perplexity"
        )
    """
    pipeline = EvaluationPipeline(model_path=model_path)
    return pipeline.run(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        evaluator_name=evaluator_name,
        **kwargs
    )


if __name__ == "__main__":
    # 示例用法
    print("\n=== Pipeline Module ===")
    print("\nAvailable datasets:")
    for name in DatasetRegistry.list_datasets():
        config = DatasetRegistry.get(name)
        print(f"  - {name}: {config.description}")
    
    print("\nAvailable evaluators:")
    for name in EvaluatorRegistry.list_evaluators():
        config = EvaluatorRegistry.get(name)
        print(f"  - {name}: {config.description}")
    
    print("\n" + "="*80)
    print("Usage Example:")
    print("="*80)
    print("""
# 方法1: 使用 Pipeline 类
pipeline = EvaluationPipeline(model_path="/path/to/model")
results = pipeline.run(
    dataset_name="lambada",
    dataset_path="/path/to/lambada.jsonl",
    evaluator_name="perplexity",
    batch_size=256
)

# 方法2: 使用 PipelineBuilder（链式调用）
results = (PipelineBuilder(model_path="/path/to/model")
    .add_task("lambada", "/path/to/lambada.jsonl", "perplexity")
    .add_task("qa", "/path/to/qa.jsonl", "exact_match")
    .run())

# 方法3: 快速评估
results = quick_eval(
    model_path="/path/to/model",
    dataset_name="lambada",
    dataset_path="/path/to/lambada.jsonl",
    evaluator_name="perplexity"
)
    """)

