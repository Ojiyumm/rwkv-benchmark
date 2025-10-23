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
        save_incorrect: bool = True,
        incorrect_only_file: Optional[str] = None,
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
            save_incorrect: 是否尝试保存所有错误案例的输入输出
            incorrect_only_file: 指定错误案例保存路径（jsonl格式），默认为输出目录下自动生成
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
        
        # 预先生成时间戳，供输出文件统一使用
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 拷贝评估器参数，避免外部字典被修改
        eval_kwargs = dict(eval_kwargs)
        
        # 如果需要收集错误样本，确保评估器可以输出逐样本结果
        inferoutput_path = eval_kwargs.get('inferoutput')
        inference_enabled = False
        if save_incorrect:
            if not inferoutput_path:
                inference_dir = os.path.join(self.output_dir, "inference_logs")
                os.makedirs(inference_dir, exist_ok=True)
                inferoutput_path = os.path.join(
                    inference_dir,
                    f"{dataset_name}_{evaluator_name}_{timestamp}.jsonl"
                )
                eval_kwargs['inferoutput'] = inferoutput_path
            # 确保目录存在
            infer_dirname = os.path.dirname(inferoutput_path)
            if infer_dirname:
                os.makedirs(infer_dirname, exist_ok=True)
            inference_enabled = True
        
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
        
        # 3. 保存结果
        print(f"\n[3/3] Finalizing results")
        
        incorrect_samples = []
        incorrect_file_path = None
        inference_file_exists = inferoutput_path and os.path.exists(inferoutput_path)
        
        if save_incorrect and inference_enabled:
            if inference_file_exists:
                with open(inferoutput_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"⚠ Skipping malformed JSON at line {line_num} in {inferoutput_path}")
                            continue
                        
                        # 仅当评估记录中提供 is_correct 字段时才判断
                        if record.get('is_correct') is False:
                            incorrect_samples.append(record)
                
                if incorrect_samples:
                    incorrect_file_path = incorrect_only_file or os.path.join(
                        self.output_dir,
                        f"{dataset_name}_{evaluator_name}_{timestamp}_incorrect.jsonl"
                    )
                    incorrect_dir = os.path.dirname(incorrect_file_path)
                    if incorrect_dir:
                        os.makedirs(incorrect_dir, exist_ok=True)
                    
                    with open(incorrect_file_path, 'w', encoding='utf-8') as f:
                        for item in incorrect_samples:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
                    print(f"✓ Incorrect samples saved to: {incorrect_file_path} (count: {len(incorrect_samples)})")
                else:
                    print("✓ No incorrect samples found; nothing saved.")
            else:
                print(f"⚠ Expected inference output at {inferoutput_path}, but file not found. Skipping incorrect sample export.")
        
        # 构建完整结果
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
        
        artifacts = {}
        if inference_enabled and inference_file_exists:
            artifacts['inference_output_file'] = inferoutput_path
        if incorrect_file_path:
            artifacts['incorrect_samples_file'] = incorrect_file_path
            artifacts['incorrect_count'] = len(incorrect_samples)
        elif save_incorrect and inference_enabled:
            artifacts['incorrect_samples_file'] = None
            artifacts['incorrect_count'] = len(incorrect_samples)
        if artifacts:
            full_results['artifacts'] = artifacts
        
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

