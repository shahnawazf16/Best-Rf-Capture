import numpy as np
from pathlib import Path
import logging
import json
import joblib
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LightSignalClassifier:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.classes: List[str] = []
        
    def extract_features(self, iq_data: np.ndarray) -> np.ndarray:
        """Extract features from IQ data for traditional ML"""
        features = []
        
        # Statistical features
        features.extend([
            np.mean(iq_data.real),
            np.mean(iq_data.imag),
            np.std(iq_data.real),
            np.std(iq_data.imag),
            np.var(iq_data.real),
            np.var(iq_data.imag),
        ])
        
        # Spectral features
        fft_mag = np.abs(np.fft.fft(iq_data))
        features.extend([
            np.mean(fft_mag),
            np.std(fft_mag),
            np.max(fft_mag),
            np.argmax(fft_mag) / len(fft_mag),  # Normalized peak frequency
        ])
        
        return np.array(features)
    
    def build_rf_model(self) -> RandomForestClassifier:
        """Build Random Forest classifier"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def build_svm_model(self) -> SVC:
        """Build SVM classifier"""
        return SVC(kernel='rbf', probability=True, random_state=42)
    
    def preprocess_iq_data(self, iq_data: np.ndarray, segment_length: int = 1024) -> np.ndarray:
        """Preprocess IQ data and extract features"""
        if len(iq_data) < segment_length:
            raise ValueError(f"Input data too short. Need at least {segment_length} samples")
            
        # Segment the data
        num_segments = len(iq_data) // segment_length
        segmented = iq_data[:num_segments * segment_length].reshape(-1, segment_length)
        
        # Extract features for each segment
        features_list = []
        for segment in segmented:
            features = self.extract_features(segment)
            features_list.append(features)
            
        return np.array(features_list)
    
    def train_model(self, data_path: Path, model_type: str = 'rf', 
                   test_size: float = 0.2) -> Dict[str, Any]:
        """Train the specified model"""
        self.logger.info(f"Training {model_type} model")
        
        # Create models directory if it doesn't exist
        Path('models').mkdir(exist_ok=True)
        
        # Create dummy data for demo
        X, y = self.create_dummy_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Build and train model
        if model_type == 'rf':
            model = self.build_rf_model()
        elif model_type == 'svm':
            model = self.build_svm_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models[model_type] = model
        self.scalers[model_type] = scaler
        
        # Save model
        self.save_model(model_type)
        
        results = {
            'accuracy': accuracy,
            'model_type': model_type
        }
        
        self.logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        return results
    
    def create_dummy_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Create dummy data for demonstration"""
        segment_length = 1024
        features_list = []
        labels = []
        
        for i in range(num_samples):
            signal_type = np.random.randint(0, 3)
            
            if signal_type == 0:
                # Pure tone
                t = np.linspace(0, 1, segment_length)
                signal = np.exp(1j * 2 * np.pi * 10 * t)
            elif signal_type == 1:
                # Noise
                signal = np.random.normal(0, 1, segment_length) + 1j * np.random.normal(0, 1, segment_length)
            else:
                # Mixed signal
                t = np.linspace(0, 1, segment_length)
                signal = np.exp(1j * 2 * np.pi * 5 * t) + 0.5 * (np.random.normal(0, 1, segment_length) + 1j * np.random.normal(0, 1, segment_length))
            
            features = self.extract_features(signal)
            features_list.append(features)
            labels.append(signal_type)
        
        self.classes = ['tone', 'noise', 'mixed']
        return np.array(features_list), np.array(labels)
    
    def classify_signals(self, input_path: Path, model_type: str = 'rf') -> List[Dict[str, Any]]:
        """Classify signals using trained model"""
        if model_type not in self.models:
            self.load_model(model_type)
            
        # Load input data
        from ..core.capture_manager import CaptureManager
        cm = CaptureManager()
        iq_data, _ = cm.load_iq_data(input_path)
        
        # Preprocess and extract features
        features = self.preprocess_iq_data(iq_data)
        
        # Scale features
        features_scaled = self.scalers[model_type].transform(features)
        
        # Predict
        predictions = self.models[model_type].predict_proba(features_scaled)
        
        # Process results
        results = []
        for i, pred in enumerate(predictions):
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            
            if len(self.classes) > class_idx:
                class_name = self.classes[class_idx]
            else:
                class_name = f"class_{class_idx}"
                
            results.append({
                'segment': i,
                'class': class_name,
                'confidence': float(confidence),
                'all_probabilities': {cls: float(prob) for cls, prob in zip(self.classes, pred)}
            })
            
        return results
    
    def save_model(self, model_type: str) -> None:
        """Save trained model"""
        model_path = f'models/{model_type}_model.joblib'
        scaler_path = f'models/{model_type}_scaler.joblib'
        classes_path = f'models/{model_type}_classes.json'
        
        joblib.dump(self.models[model_type], model_path)
        joblib.dump(self.scalers[model_type], scaler_path)
        
        with open(classes_path, 'w') as f:
            json.dump(self.classes, f)
        
        self.logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_type: str) -> None:
        """Load pre-trained model"""
        model_path = f'models/{model_type}_model.joblib'
        scaler_path = f'models/{model_type}_scaler.joblib'
        classes_path = f'models/{model_type}_classes.json'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.models[model_type] = joblib.load(model_path)
        self.scalers[model_type] = joblib.load(scaler_path)
        
        if Path(classes_path).exists():
            with open(classes_path, 'r') as f:
                self.classes = json.load(f)
        
        self.logger.info(f"Model loaded from {model_path}")