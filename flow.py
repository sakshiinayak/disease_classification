import mlflow
import mlflow.keras
import os
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
import traceback
load_dotenv()
class MLflowConnectionChecker:
    def __init__(self, mlflow_uri=None):
        """
        Initialize MLflow connection checker
        
        Args:
            mlflow_uri: MLflow tracking URI (e.g., "https://dagshub.com/username/repo.mlflow")
        """
        self.mlflow_uri = mlflow_uri
        load_dotenv()  # Load environment variables from .env file
        
    def setup_credentials(self, username=None, password=None):
        """
        Set up MLflow credentials
        
        Args:
            username: MLflow tracking username
            password: MLflow tracking password/token
        """
        # Try to get from parameters first, then environment variables
        username = username or os.getenv("MLFLOW_TRACKING_USERNAME")
        password = password or os.getenv("MLFLOW_TRACKING_PASSWORD")
        
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
            print(f"✅ Credentials set for user: {username}")
        else:
            print("⚠️  No credentials found. Please set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD")
            
    def check_basic_connection(self):
        """Check basic MLflow connection"""
        print("\n🔍 Checking basic MLflow connection...")
        
        try:
            if self.mlflow_uri:
                mlflow.set_tracking_uri(self.mlflow_uri)
                
            current_uri = mlflow.get_tracking_uri()
            print(f"✅ MLflow tracking URI: {current_uri}")
            
            # Check URL scheme
            parsed_uri = urlparse(current_uri)
            print(f"✅ URL scheme: {parsed_uri.scheme}")
            print(f"✅ URL netloc: {parsed_uri.netloc}")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic connection failed: {e}")
            return False
    
    def check_experiments_access(self):
        """Check if we can access experiments"""
        print("\n🔍 Checking experiments access...")
        
        try:
            experiments = mlflow.search_experiments()
            print(f"✅ Successfully accessed experiments")
            print(f"✅ Found {len(experiments)} experiments")
            
            if experiments:
                print("📋 Available experiments:")
                for exp in experiments[:5]:  # Show first 5 experiments
                    print(f"   - {exp.name} (ID: {exp.experiment_id})")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to access experiments: {e}")
            print(f"🔍 Error details: {traceback.format_exc()}")
            return False
    
    def check_http_connection(self):
        """Check HTTP connection to MLflow server"""
        print("\n🔍 Checking HTTP connection...")
        
        try:
            if not self.mlflow_uri:
                print("⚠️  No MLflow URI provided, skipping HTTP check")
                return False
                
            # Parse the URI to get the base URL
            parsed_uri = urlparse(self.mlflow_uri)
            base_url = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
            
            # Try to access the MLflow API endpoint
            auth = None
            username = os.getenv("MLFLOW_TRACKING_USERNAME")
            password = os.getenv("MLFLOW_TRACKING_PASSWORD")
            
            if username and password:
                auth = (username, password)
            
            response = requests.get(f"{base_url}/api/2.0/mlflow/experiments/search", 
                                  auth=auth, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ HTTP connection successful (Status: {response.status_code})")
                return True
            else:
                print(f"❌ HTTP connection failed (Status: {response.status_code})")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ HTTP connection failed: {e}")
            return False
    
    def test_create_run(self):
        """Test creating a simple MLflow run"""
        print("\n🔍 Testing MLflow run creation...")
        
        try:
            # End any existing active run
            if mlflow.active_run():
                print("🔄 Ending existing active run...")
                mlflow.end_run()
            
            # Create a test run
            with mlflow.start_run() as run:
                print(f"✅ Successfully created run: {run.info.run_id}")
                
                # Log test parameters
                mlflow.log_param("test_param", "test_value")
                print("✅ Successfully logged parameter")
                
                # Log test metrics
                mlflow.log_metric("test_metric", 0.95)
                print("✅ Successfully logged metric")
                
                # Log test artifact
                with open("test_artifact.txt", "w") as f:
                    f.write("This is a test artifact")
                mlflow.log_artifact("test_artifact.txt")
                print("✅ Successfully logged artifact")
                
                print(f"✅ Run completed successfully!")
                print(f"Run ID: {run.info.run_id}")
                
            # Clean up
            if os.path.exists("test_artifact.txt"):
                os.remove("test_artifact.txt")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to create test run: {e}")
            print(f"🔍 Error details: {traceback.format_exc()}")
            
            # Ensure we end the run even if there's an error
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
                
            return False
    
    def run_full_check(self, username=None, password=None):
        """Run all connection checks"""
        print("🚀 Starting MLflow connection check...")
        print("="*50)
        
        # Setup credentials
        self.setup_credentials(username, password)
        
        # Run all checks
        checks = [
            ("Basic Connection", self.check_basic_connection),
            ("HTTP Connection", self.check_http_connection),
            ("Experiments Access", self.check_experiments_access),
            ("Test Run Creation", self.test_create_run)
        ]
        
        results = {}
        for check_name, check_func in checks:
            results[check_name] = check_func()
        
        # Summary
        print("\n📊 CONNECTION CHECK SUMMARY")
        print("="*50)
        for check_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{check_name}: {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\n🎉 All checks passed! MLflow connection is working properly.")
        else:
            print("\n⚠️  Some checks failed. Please review the errors above.")
            
        return all_passed

# Example usage for DagsHub
def check_dagshub_connection():
    """Specific function to check DagsHub MLflow connection using environment variables"""
    
    # Get credentials from environment variables
    DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
    DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
    
    if not all([DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN]):
        print("❌ Missing required environment variables:")
        print("   - DAGSHUB_USERNAME")
        print("   - DAGSHUB_REPO")
        print("   - DAGSHUB_TOKEN")
        print("Please set these in your .env file")
        return False
    
    mlflow_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    
    checker = MLflowConnectionChecker(mlflow_uri)
    return checker.run_full_check(username=DAGSHUB_USERNAME, password=DAGSHUB_TOKEN)

# Quick connection test function
def quick_mlflow_test(mlflow_uri, username=None, password=None):
    """Quick test for MLflow connection"""
    try:
        # Set credentials if provided
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        
        # Set tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Test basic connection
        experiments = mlflow.search_experiments()
        print(f"✅ Connection successful! Found {len(experiments)} experiments")
        
        # Test creating a run
        with mlflow.start_run() as run:
            mlflow.log_metric("test", 1.0)
            print(f"✅ Test run created: {run.info.run_id}")
            
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    print("MLflow Connection Checker")
    print("="*30)
    
    # Option 1: Check DagsHub connection using environment variables
    print("\n1. Checking DagsHub connection using environment variables...")
    check_dagshub_connection()
    
    # Option 2: Manual check with environment variables
    print("\n2. Manual connection check using environment variables...")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME") 
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if mlflow_uri and username and password:
        checker = MLflowConnectionChecker(mlflow_uri)
        checker.run_full_check(username, password)
    else:
        print("❌ Missing environment variables: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD")
    
    # Option 3: Quick test with environment variables
    print("\n3. Quick connection test using environment variables...")
    if mlflow_uri and username and password:
        quick_mlflow_test(mlflow_uri, username, password)
    else:
        print("❌ Missing environment variables for quick test")