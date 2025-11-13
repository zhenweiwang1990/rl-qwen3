#!/usr/bin/env python3
"""Verify training database setup."""

import os
import csv
from pathlib import Path

from tools import DEFAULT_DB_PATH

def main():
    """Verify setup."""
    script_dir = Path(__file__).parent
    
    print('='*80)
    print('训练数据库配置验证')
    print('='*80)
    print()
    
    # Check database
    print(f'✅ 默认数据库路径: {DEFAULT_DB_PATH}')
    db_exists = os.path.exists(DEFAULT_DB_PATH)
    print(f'✅ 数据库存在: {db_exists}')
    
    if db_exists:
        db_size = Path(DEFAULT_DB_PATH).stat().st_size / (1024 * 1024)
        print(f'✅ 数据库大小: {db_size:.1f} MB')
    print()
    
    # Check training benchmark
    training_benchmark = script_dir / "data" / "benchmark-queries-training.csv"
    if training_benchmark.exists():
        with open(training_benchmark, 'r') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
        print(f'✅ 训练 Benchmark 存在: {count} 个查询')
    else:
        print(f'⚠️  训练 Benchmark 未找到: {training_benchmark}')
    print()
    
    # Check full benchmark
    full_benchmark = script_dir / "data" / "benchmark-queries-flattened.csv"
    if full_benchmark.exists():
        with open(full_benchmark, 'r') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
        print(f'✅ 完整 Benchmark 存在: {count} 个查询')
    else:
        print(f'⚠️  完整 Benchmark 未找到: {full_benchmark}')
    print()
    
    # Check profile list
    profile_list = script_dir / "data" / "10000-training-linkedin-handle.csv"
    if profile_list.exists():
        with open(profile_list, 'r') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
        print(f'✅ Profile 列表存在: {count} 个 handles')
    else:
        print(f'⚠️  Profile 列表未找到: {profile_list}')
    print()
    
    print('='*80)
    if db_exists and training_benchmark.exists():
        print('✅ 所有配置正确！训练数据库系统已准备就绪。')
        print()
        print('下一步:')
        print('  - 运行测试: python3 test_training_db.py')
        print('  - 运行 benchmark: python3 benchmark.py -n 10')
        print('  - 启动 CLI: python3 cli.py')
    else:
        print('⚠️  设置未完成，请运行: ./setup_training_db.sh')
    print('='*80)

if __name__ == "__main__":
    main()

