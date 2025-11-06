#!/bin/bash
set -e

echo "=========================================="
echo "Compare Multiple Models"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./scripts/setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create one based on env.example"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Set defaults
export TEST_SET_SIZE="${TEST_SET_SIZE:-100}"

# Models to compare
MODELS=(
    "gpt-4o:openai/gpt-4o:true"
    "gemini-2.0-flash:gemini/gemini-2.0-flash:false"
    "qwen3-email-agent-001::true"
)

echo "Configuration:"
echo "  - Test set size: $TEST_SET_SIZE"
echo "  - Models to compare: ${#MODELS[@]}"
echo ""

# Create comparison script
cat > /tmp/compare_models.py << 'EOF'
import asyncio
import art
from dotenv import load_dotenv
import os
import sys
from qwen3_agent.config import PolicyConfig
from qwen3_agent.benchmark import benchmark_model
import polars as pl
from tabulate import tabulate

load_dotenv()

async def main():
    # Parse models from environment
    models_str = os.environ.get("MODELS_TO_COMPARE", "")
    models_list = [m.strip() for m in models_str.split(";") if m.strip()]
    
    test_set_size = int(os.environ.get("TEST_SET_SIZE", "100"))
    
    api = art.LocalAPI()
    models = []
    model_names = []
    
    for model_config in models_list:
        parts = model_config.split(":")
        if len(parts) < 3:
            continue
            
        model_name, litellm_name, use_tools = parts[0], parts[1], parts[2] == "true"
        model_names.append(model_name)
        
        if litellm_name:
            # External model
            model = art.Model(
                name=model_name,
                project="qwen3_email_agent",
                inference_api_key=os.getenv("OPENAI_API_KEY"),
                inference_base_url="https://api.openai.com/v1",
                config=PolicyConfig(
                    litellm_model_name=litellm_name,
                    use_tools=use_tools,
                ),
            )
        else:
            # Our trained model
            model = art.Model(
                name=model_name,
                project="qwen3_email_agent",
                config=PolicyConfig(use_tools=use_tools),
            )
            
            # Try to pull from S3
            if os.environ.get("BACKUP_BUCKET"):
                try:
                    await api._experimental_pull_from_s3(
                        model,
                        s3_bucket=os.environ["BACKUP_BUCKET"],
                        verbose=False,
                    )
                except Exception as e:
                    print(f"Could not pull {model_name} from S3: {e}")
        
        await model.register(api)
        models.append(model)
        print(f"✓ Configured {model_name}")
    
    print(f"\nRunning benchmarks on {test_set_size} test scenarios...")
    print("")
    
    # Run benchmarks in parallel
    results = await asyncio.gather(
        *[benchmark_model(model, test_set_size) for model in models]
    )
    
    # Combine results
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80 + "\n")
    
    # Create comparison table
    all_metrics = {}
    for model_name, result in zip(model_names, results):
        metrics = result.to_dict()
        for key in metrics:
            if key not in all_metrics:
                all_metrics[key] = {}
            all_metrics[key][model_name] = metrics[key][0]
    
    # Print as table
    headers = ["Metric"] + model_names
    rows = []
    for metric, values in all_metrics.items():
        row = [metric] + [f"{values.get(model, 0):.4f}" for model in model_names]
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Save to CSV
    output_file = "model_comparison.csv"
    df = pl.DataFrame(all_metrics)
    df.write_csv(output_file)
    print(f"\n✓ Results saved to {output_file}")

asyncio.run(main())
EOF

# Convert models array to environment variable
MODELS_STR=""
for model in "${MODELS[@]}"; do
    MODELS_STR="${MODELS_STR}${model};"
done
export MODELS_TO_COMPARE="${MODELS_STR}"

# Run comparison
python /tmp/compare_models.py

# Cleanup
rm /tmp/compare_models.py

echo ""
echo "=========================================="
echo "Model Comparison Complete!"
echo "=========================================="

