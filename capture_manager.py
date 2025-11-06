import numpy as np
import logging
from pathlib import Path
import h5py

class CaptureManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def capture_to_file(self, sdr, duration: float, output_path: Path):
        """Capture samples and save to file"""
        sample_rate = sdr.config['sample_rate']
        total_samples = int(sample_rate * duration)
        
        self.logger.info(f"Capturing {total_samples} samples...")
        
        try:
            # For demo purposes, create dummy data
            # In real implementation, use sdr.capture_samples()
            iq_data = self.create_dummy_signal(total_samples)
            
            # Save to file
            self.save_iq_data(iq_data, output_path, sample_rate)
            self.logger.info(f"Capture completed. Saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Capture failed: {e}")
            raise
            
    def create_dummy_signal(self, num_samples: int) -> np.ndarray:
        """Create dummy RF signal for testing"""
        t = np.linspace(0, 1, num_samples)
        # Create a simple FSK-like signal
        carrier = np.exp(1j * 2 * np.pi * 100 * t)
        modulation = np.sin(2 * np.pi * 10 * t)
        signal = carrier * (1 + 0.5 * modulation)
        
        # Add some noise
        noise = 0.1 * (np.random.normal(0, 1, num_samples) + 1j * np.random.normal(0, 1, num_samples))
        return signal + noise
            
    def save_iq_data(self, iq_data: np.ndarray, output_path: Path, sample_rate: float):
        """Save IQ data to file"""
        output_path = Path(output_path)
        
        if output_path.suffix == '.npy':
            np.save(output_path, iq_data)
        elif output_path.suffix == '.h5':
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('iq_data', data=iq_data)
                f.attrs['sample_rate'] = sample_rate
                f.attrs['samples'] = len(iq_data)
        else:
            # Default to numpy format
            np.save(output_path.with_suffix('.npy'), iq_data)
            
    def load_iq_data(self, input_path: Path) -> tuple:
        """Load IQ data from file"""
        input_path = Path(input_path)
        
        if input_path.suffix == '.npy':
            iq_data = np.load(input_path)
            sample_rate = None  # Unknown for npy files
        elif input_path.suffix == '.h5':
            with h5py.File(input_path, 'r') as f:
                iq_data = f['iq_data'][:]
                sample_rate = f.attrs.get('sample_rate', None)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
            
        return iq_data, sample_rate