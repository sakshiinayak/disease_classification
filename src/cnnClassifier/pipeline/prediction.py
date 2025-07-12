import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        try:
            # Get project root directory and construct correct model path
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = project_root / "model" / "model.h5"
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            print(f"Loading model from: {model_path}")
            
            # Load model
            model = load_model(str(model_path))
            
            # Check if image file exists
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"Image file not found: {self.filename}")
            
            print(f"Processing image: {self.filename}")
            
            # Load and preprocess image
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0  # Normalize
            
            # Make prediction
            prediction_probs = model.predict(test_image, verbose=0)
            
            print(f"Raw prediction: {prediction_probs}")
            print(f"Prediction shape: {prediction_probs.shape}")
            
            # Handle different model outputs
            if prediction_probs.shape[1] == 1:
                # Single output (sigmoid) - binary classification
                prob = prediction_probs[0][0]
                print(f"Single output value: {prob:.4f}")
                
                # Use confidence-based logic
                confidence_threshold = 0.9  # 90% confidence threshold
                
                if prob > confidence_threshold:
                    prediction = 'Normal'
                    confidence = prob * 100
                elif prob < (1 - confidence_threshold):  # Less than 10% (high confidence for other class)
                    prediction = 'Normal'
                    confidence = (1 - prob) * 100
                else:
                    # Low confidence - predict as Tumor for safety
                    prediction = 'Tumor'
                    confidence = max(prob, 1 - prob) * 100
                    
            else:
                # Multi-class output (softmax)
                class_0_prob = prediction_probs[0][0]
                class_1_prob = prediction_probs[0][1]
                
                print(f"Class 0 probability: {class_0_prob:.4f}")
                print(f"Class 1 probability: {class_1_prob:.4f}")
                
                # Get the maximum confidence
                max_confidence = max(class_0_prob, class_1_prob)
                confidence = max_confidence * 100
                
                print(f"Max confidence: {confidence:.2f}%")
                
                # Conservative approach: if confidence < 90%, predict Tumor
                if confidence < 90.0:
                    prediction = 'Tumor'
                    print("Low confidence - predicting Tumor for safety")
                else:
                    # High confidence - use the actual prediction
                    if class_0_prob > class_1_prob:
                        prediction = 'Normal'
                    else:
                        prediction = 'Tumor'
                    print(f"High confidence - using model prediction")
                
                print(f"Decision logic: Confidence {confidence:.2f}% -> {prediction}")
            
            print(f"Final prediction: {prediction}")
            print(f"Confidence: {confidence:.2f}%")
            
            return [{"image": prediction, "confidence": f"{confidence:.2f}%"}]
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return [{"error": str(e)}]