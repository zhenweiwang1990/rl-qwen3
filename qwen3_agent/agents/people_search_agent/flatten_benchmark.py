#!/usr/bin/env python3
"""
将 benchmark-queries.csv 中的 answer 字段里的 content 字段合并成每个问题一个数组。
去掉 batch 层级，直接拍平所有 content。
"""

import csv
import json
from pathlib import Path


def flatten_answers(input_file: str, output_file: str):
    """
    读取 benchmark-queries.csv，将 answer 中的所有 content 拍平成一个数组。
    
    Args:
        input_file: 输入的 CSV 文件路径
        output_file: 输出的 CSV 文件路径
    """
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        
        # 准备输出数据
        output_rows = []
        
        for row in reader:
            question_id = row['id']
            query = row['query']
            answer_json = row['answer']
            batch = row['batch']
            updated_at = row['updated_at']
            
            # 解析 answer JSON
            try:
                answer_data = json.loads(answer_json)
                
                # 拍平所有 batch 的 content
                flattened_content = []
                for item in answer_data:
                    if 'content' in item:
                        flattened_content.extend(item['content'])
                
                # 保存结果
                output_rows.append({
                    'id': question_id,
                    'query': query,
                    'content': json.dumps(flattened_content, ensure_ascii=False),
                    'content_count': len(flattened_content),
                    'batch': batch,
                    'updated_at': updated_at
                })
                
            except json.JSONDecodeError as e:
                print(f"警告: ID {question_id} 的 answer 字段解析失败: {e}")
                output_rows.append({
                    'id': question_id,
                    'query': query,
                    'content': '[]',
                    'content_count': 0,
                    'batch': batch,
                    'updated_at': updated_at
                })
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            fieldnames = ['id', 'query', 'content', 'content_count', 'batch', 'updated_at']
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(output_rows)
    
    print(f"✅ 处理完成!")
    print(f"   输入文件: {input_file}")
    print(f"   输出文件: {output_file}")
    print(f"   处理了 {len(output_rows)} 个问题")


def main():
    # 文件路径
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'benchmark-queries.csv'
    output_file = data_dir / 'benchmark-queries-flattened.csv'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"❌ 错误: 输入文件不存在 - {input_file}")
        return
    
    # 执行拍平操作
    flatten_answers(str(input_file), str(output_file))
    
    # 显示示例
    print("\n📊 输出示例 (前3行):")
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 3:
                break
            content = json.loads(row['content'])
            print(f"\nID: {row['id']}")
            print(f"Query: {row['query'][:80]}...")
            print(f"Content Count: {row['content_count']}")
            print(f"First 5 items: {content[:5]}")


if __name__ == '__main__':
    main()

