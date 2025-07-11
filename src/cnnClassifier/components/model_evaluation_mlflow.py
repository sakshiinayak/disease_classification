import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        # Load environment variables from .env file
        load_dotenv()
        
        # Debug: Print the actual config being used
        print(f"Debug - Config training_data path: {self.config.training_data}")
        print(f"Debug - Config model_path: {self.config.model_path}")

    def _validate_paths(self):
        """Validate that all required paths exist before proceeding"""
        
        # Check training data path
        training_data_path = Path(self.config.training_data)
        if not training_data_path.exists():
            print(f"Training data path does not exist: {training_data_path}")
            
            # Show what actually exists
            base_path = Path("artifacts/data_ingestion")
            if base_path.exists():
                print(f"Available directories in {base_path}:")
                for item in base_path.iterdir():
                    if item.is_dir():
                        print(f" {item.name}")
            
            raise FileNotFoundError(
                f"Training data directory not found: {training_data_path}\n"
                f"Please check your configuration."
            )
        
        # Check model path
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please ensure the model training stage has been completed successfully."
            )
        
        # Check if training data has class subdirectories
        class_dirs = [d for d in training_data_path.iterdir() if d.is_dir()]
        if len(class_dirs) == 0:
            raise ValueError(
                f"No class directories found in: {training_data_path}\n"
                f"Expected structure: {training_data_path}/[class_name]/[images]"
            )
        
        print(f"  - Training data: {training_data_path}")
        print(f"  - Model file: {model_path}")
        print(f"  - Classes found: {[d.name for d in class_dirs]}")
        
        return True

    def _valid_generator(self):
        """Create validation data generator"""
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        print(f"Creating data generator from: {self.config.training_data}")
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Use the config path directly
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """Run model evaluation with proper validation"""
        try:
            print("Starting model evaluation")
            
            # Validate paths first
            self._validate_paths()
            
            # Load model
            self.model = self.load_model(self.config.model_path)
            print("✅ Model loaded successfully")
            
            # Create validation generator
            self._valid_generator()
            print("Validation generator created successfully")
            
            # Evaluate model
            self.score = self.model.evaluate(self.valid_generator)
            print(f" Evaluation completed!")
            print(f"   Loss: {self.score[0]:.4f}")
            print(f"   Accuracy: {self.score[1]:.4f}")
            
            # Save scores
            self.save_score()
            print("Scores saved to scores.json")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def test_mlflow_connection(self):
        """Test MLflow connection to DagsHub using environment variables"""
        try:
            # Set MLflow tracking URI from environment
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            
            # Test basic connection
            experiments = mlflow.search_experiments()
            print(f"✅ Connected to DagsHub MLflow successfully!")
            print(f"Found {len(experiments)} experiments")
            print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            
            # Test creating a simple run
            with mlflow.start_run():
                mlflow.log_metric("test_metric", 1.0)
                print("Test run created successfully!")
                
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
        return True

    def log_into_mlflow(self):
        """Log model metrics and artifacts to MLflow (DagsHub) using environment variables"""
        try:
            # MLflow will automatically use environment variables:
            # MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            
            # End any existing active run
            if mlflow.active_run():
                print("Ending existing active run...")
                mlflow.end_run()
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Tracking URL type: {tracking_url_type_store}")
            
            # Test connection before starting run
            try:
                experiments = mlflow.search_experiments()
                print(f"Successfully connected to MLflow. Found {len(experiments)} experiments.")
            except Exception as e:
                print(f"Failed to connect to MLflow: {e}")
                return

            # Start MLflow run
            with mlflow.start_run() as run:
                print(f"Started new MLflow run: {run.info.run_id}")
                
                # Log parameters
                mlflow.log_params(self.config.all_params)
                print("Parameters logged successfully")
                
                # Log metrics
                mlflow.log_metrics({
                    "loss": self.score[0],
                    "accuracy": self.score[1]
                })
                print(f"Metrics logged - Loss: {self.score[0]:.4f}, Accuracy: {self.score[1]:.4f}")

                # Log model - always log for DagsHub
                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    keras_model_kwargs={"save_format": "h5"},
                    registered_model_name="VGG16Model"
                )
                print("Model logged successfully")
                
                print(f"MLflow run completed successfully!")
                print(f"Run ID: {run.info.run_id}")
                print(f"View your experiment at: {self.config.mlflow_uri}")
                
        except Exception as e:
            print(f"Error logging to MLflow: {e}")
            # End the run with failure status if there's an error
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
            raise

    def verify_env_variables(self):
        """Verify that all required environment variables are set"""
        required_vars = [
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME", 
            "MLFLOW_TRACKING_PASSWORD",
            "DAGSHUB_USERNAME",
            "DAGSHUB_REPO",
            "DAGSHUB_TOKEN"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"Missing environment variables: {missing_vars}")
            return False
        else:
            print("All required environment variables are set")
            return True