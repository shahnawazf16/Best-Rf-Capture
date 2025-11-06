import numpy as np
import logging
from typing import Optional, Dict, Any, Callable

class SDRController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.config: Dict[str, Any] = {}
        
    def configure(self, frequency: float, sample_rate: float, 
                 gain: float, device_type: str = 'rtlsdr') -> None:
        """Configure SDR device"""
        self.config.update({
            'frequency': frequency,
            'sample_rate': sample_rate,
            'gain': gain,
            'device_type': device_type
        })
        
        try:
            if device_type == 'rtlsdr':
                self._init_rtlsdr()
            elif device_type == 'hackrf':
                self._init_hackrf()
            elif device_type == 'usrp':
                self._init_usrp()
            else:
                raise ValueError(f"Unsupported device type: {device_type}")
                
        except ImportError as e:
            self.logger.error(f"SDR driver not available: {e}")
            raise
            
    def _init_rtlsdr(self) -> None:
        """Initialize RTL-SDR device"""
        try:
            from rtlsdr import RtlSdr
            self.device = RtlSdr()
            self.device.set_center_freq(int(self.config['frequency']))
            self.device.set_sample_rate(int(self.config['sample_rate']))
            self.device.set_gain(self.config['gain'])
            self.logger.info("RTL-SDR initialized successfully")
        except Exception as e:
            self.logger.error(f"RTL-SDR initialization failed: {e}")
            raise
            
    def _init_hackrf(self) -> None:
        """Initialize HackRF device"""
        try:
            self.logger.warning("HackRF support requires additional drivers")
            raise ImportError("HackRF driver not installed")
        except Exception as e:
            self.logger.error(f"HackRF initialization failed: {e}")
            raise
            
    def _init_usrp(self) -> None:
        """Initialize USRP device"""
        try:
            self.logger.warning("USRP support requires UHD drivers")
            raise ImportError("USRP driver not installed")
        except Exception as e:
            self.logger.error(f"USRP initialization failed: {e}")
            raise
            
    def capture_samples(self, num_samples: int) -> np.ndarray:
        """Capture IQ samples from SDR"""
        if not self.device:
            raise RuntimeError("SDR device not initialized")
            
        try:
            if hasattr(self.device, 'read_samples'):
                samples = self.device.read_samples(num_samples)
                return np.array(samples, dtype=np.complex64)
            else:
                # For testing, return dummy data
                return self._create_dummy_samples(num_samples)
                
        except Exception as e:
            self.logger.error(f"Capture failed: {e}")
            # Return dummy data for testing
            return self._create_dummy_samples(num_samples)
            
    def _create_dummy_samples(self, num_samples: int) -> np.ndarray:
        """Create dummy samples for testing without hardware"""
        t = np.linspace(0, 1, num_samples)
        carrier = np.exp(1j * 2 * np.pi * 100 * t)
        modulation = np.sin(2 * np.pi * 10 * t)
        signal = carrier * (1 + 0.5 * modulation)
        noise = 0.1 * (np.random.normal(0, 1, num_samples) + 1j * np.random.normal(0, 1, num_samples))
        return signal + noise
            
    def start_streaming(self, callback: Callable, num_samples: int = 1024*1024) -> None:
        """Start continuous streaming with callback"""
        if hasattr(self.device, 'read_samples_async'):
            self.device.read_samples_async(callback, num_samples=num_samples)
        else:
            self.logger.warning("Streaming not supported for this device")
            
    def close(self) -> None:
        """Close SDR device"""
        if self.device:
            if hasattr(self.device, 'close'):
                self.device.close()
            elif hasattr(self.device, 'cancel_read_async'):
                self.device.cancel_read_async()
            self.device = None