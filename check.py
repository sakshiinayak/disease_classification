from cnnClassifier.utils.common import read_yaml
from pathlib import Path

def debug_config():
    """Debug configuration loading to see what paths are being used"""
    
    # Read configuration files
    config_path = Path("config/config.yaml")
    params_path = Path("params.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    if not params_path.exists():
        print(f"❌ Params file not found: {params_path}")
        return
    
    # Load configs
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    print("📋 Configuration loaded successfully")
    print("\n🔍 Evaluation Configuration:")
    print(f"  Model Path: {config.evaluation.model_path}")
    print(f"  Training Data: {config.evaluation.training_data}")
    print(f"  MLflow URI: {config.evaluation.mlflow_uri}")
    
    print("\n📁 Path Validation:")
    model_path = Path(config.evaluation.model_path)
    training_data_path = Path(config.evaluation.training_data)
    
    print(f"  Model exists: {'✅' if model_path.exists() else '❌'} - {model_path}")
    print(f"  Training data exists: {'✅' if training_data_path.exists() else '❌'} - {training_data_path}")
    
    if training_data_path.exists():
        classes = [d.name for d in training_data_path.iterdir() if d.is_dir()]
        print(f"  Classes found: {classes}")

if __name__ == "__main__":
    debug_config()