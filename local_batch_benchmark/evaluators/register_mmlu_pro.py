"""
注册 MMLU Pro 评估器
"""

import json
from typing import List, Dict
from tqdm import tqdm
from .evaluator_registry import EvaluatorRegistry


def mmlu_pro_evaluator(
    engine,
    data: List[Dict],
    batch_size: int = 64,
    max_length: int = 1,  # MMLU-Pro 只需要生成一个字母
    use_cot: bool = False,
    inferoutput: str = None,  # 保存推理结果的路径
    **kwargs
) -> Dict[str, float]:
    """
    MMLU Pro 评估器
    评估多选题的准确率
    """
    print(f"\n=== Running MMLU Pro Evaluation ===")
    print(f"Total samples: {len(data)}")
    print(f"Batch size: {batch_size}")
    
    cot_detected = use_cot or any(
        "<think>" in item.get("prompt", "")
        for item in data[: min(len(data), 8)]
    )
    use_cot = cot_detected
    print(f"Use CoT: {use_cot}")
    if inferoutput:
        print(f"Inference output will be saved to: {inferoutput}\n")
    else:
        print()
    
    correct = 0
    total = 0
    subject_stats = {}  # 按学科统计
    inference_results = []  # 保存推理结果
    
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # 批量处理 - 使用 tqdm 显示进度
    num_batches = (len(data) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(data), desc="Evaluating MMLU-Pro", unit="samples")
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        
        # 生成
        tokens, _ = engine.generate_batch(prompts, max_length=max_length)
        predictions = engine.decode_tokens(tokens)
        
        # 评估每个样本
        for j, (pred, item) in enumerate(zip(predictions, batch)):
            # 提取预测答案（找到第一个 A-J 字母）
            pred_answer = None
            pred_clean = pred.strip().upper()
            
            for char in pred_clean:
                if char in option_labels:
                    pred_answer = char
                    break
            
            # 获取正确答案（已经在 dataset 加载时提取到 reference）
            reference = item['reference']
            
            # 标准化答案格式
            if isinstance(reference, int):
                # 如果答案是索引
                correct_answer = option_labels[reference]
            elif isinstance(reference, str):
                # 如果答案是字母或文本
                if reference.upper() in option_labels:
                    correct_answer = reference.upper()
                else:
                    # 尝试从文本中提取字母
                    for char in reference.upper():
                        if char in option_labels:
                            correct_answer = char
                            break
                    else:
                        correct_answer = reference.upper()
            else:
                correct_answer = str(reference).upper()
            
            # 判断是否正确
            is_correct = (pred_answer == correct_answer)
            if is_correct:
                correct += 1
            
            total += 1
            
            # 按学科统计（支持 category 或 subject 字段）
            subject = item['raw'].get('category', item['raw'].get('subject', 'unknown'))
            if subject not in subject_stats:
                subject_stats[subject] = {'correct': 0, 'total': 0}
            subject_stats[subject]['total'] += 1
            if is_correct:
                subject_stats[subject]['correct'] += 1
            
            # 保存推理结果（如果需要）
            if inferoutput:
                inference_results.append({
                    'question': item['raw'].get('question', ''),
                    'options': item['raw'].get('options', []),
                    'category': subject,
                    'prompt': item['prompt'],
                    'prediction': pred,
                    'predicted_answer': pred_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct
                })
        
        # 更新进度条
        pbar.update(len(batch))
        pbar.set_postfix({'accuracy': f'{correct/total*100:.2f}%'})
    
    pbar.close()
    
    # 保存推理结果到 jsonl 文件
    if inferoutput:
        import os
        os.makedirs(os.path.dirname(inferoutput) if os.path.dirname(inferoutput) else '.', exist_ok=True)
        with open(inferoutput, 'w', encoding='utf-8') as f:
            for result in inference_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\n✓ Inference results saved to: {inferoutput}")
    
    # 计算总体准确率
    accuracy = correct / total if total > 0 else 0
    
    # 计算各学科准确率
    subject_accuracies = {}
    for subject, stats in subject_stats.items():
        subject_accuracies[subject] = stats['correct'] / stats['total']
    
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'subject_accuracies': subject_accuracies,
        'num_subjects': len(subject_stats)
    }
    
    # 打印详细结果
    print(f"\n=== Final Results ===")
    print(f"Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"\nBy Subject:")
    for subject, acc in sorted(subject_accuracies.items(), key=lambda x: x[1], reverse=True):
        stats = subject_stats[subject]
        print(f"  {subject:30s}: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    return results


# 注册评估器
EvaluatorRegistry.register(
    name='mmlu_pro',
    evaluator=mmlu_pro_evaluator,
    description='MMLU Pro evaluator with subject-wise statistics',
    metrics=['accuracy', 'correct', 'total', 'subject_accuracies', 'num_subjects']
)

