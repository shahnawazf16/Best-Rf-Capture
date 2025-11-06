import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional, Any

class SignalClassifier:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, tf.keras.Model] = {}
        self.classes: List[str] = []
        
    def build_cnn_model(self, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """Build CNN model for signal classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_iq_data(self, iq_data: np.ndarray, segment_length: int = 1024) -> np.ndarray:
        """Preprocess IQ data for ML models"""
        if len(iq_data) < segment_length:
            raise ValueError(f"Input data too short. Need at least {segment_length} samples")
            
        # Segment the data
        num_segments = len(iq_data) // segment_length
        if num_segments == 0:
            raise ValueError("No complete segments can be formed")
            
        segmented = iq_data[:num_segments * segment_length].reshape(-1, segment_length)
        
        # Normalize each segment
        segmented_normalized = np.zeros_like(segmented)
        for i in range(len(segmented)):
            segment = segmented[i]
            segmented_normalized[i] = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        
        return segmented_normalized
    
    def train_model(self, data_path: Path, model_type: str = 'cnn', 
                   epochs: int = 100, batch_size: int = 32) -> Any:
        """Train the specified model"""
        self.logger.info(f"Training {model_type} model")
        
        # Create models directory if it doesn't exist
        Path('models').mkdir(exist_ok=True)
        
        # For demo purposes, create dummy data
        X_train, X_val, y_train, y_val = self.create_dummy_data()
        
        # Build model
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        
        if model_type == 'cnn':
            model = self.build_cnn_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models[model_type] = model
        self.save_model(model, model_type)
        
        return history
    
    def create_dummy_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create dummy data for demonstration"""
        # Generate synthetic RF signals for demo
        num_samples = 1000
        segment_length = 1024
        
        # Create different signal types
        signals = []
        labels = []
        
        for i in range(num_samples):
            # Random signal type
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
            
            signals.append(signal)
            labels.append(signal_type)
        
        X = np.array(signals)
        y = tf.keras.utils.to_categorical(labels, 3)
        
        # Split train/validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.classes = ['tone', 'noise', 'mixed']
        
        return X_train, X_val, y_train, y_val
    
    def classify_signals(self, input_path: Path, model_type: str = 'cnn') -> List[Dict[str, Any]]:
        """Classify signals using trained model"""
        if model_type not in self.models:
            self.load_model(model_type)
            
        # Load input data
        from ..core.capture_manager import CaptureManager
        cm = CaptureManager()
        iq_data, _ = cm.load_iq_data(input_path)
        
        # Preprocess
        processed_data = self.preprocess_iq_data(iq_data)
        
        # Predict
        predictions = self.models[model_type].predict(processed_data)
        
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
    
    def save_model(self, model: tf.keras.Model, model_type: str) -> None:
        """Save trained model"""
        model_path = f'models/{model_type}_model.h5'
        model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Save class labels
        with open(f'models/{model_type}_classes.json', 'w') as f:
            json.dump(self.classes, f)
        
    def load_model(self, model_type: str) -> None:
        """Load pre-trained model"""
        model_path = f'models/{model_type}_model.h5'
        classes_path = f'models/{model_type}_classes.json'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.models[model_type] = tf.keras.models.load_model(model_path)
        
        if Path(classes_path).exists():
            with open(classes_path, 'r') as f:
                self.classes = json.load(f)
        
        self.logger.info(f"Model loaded from {model_path}")