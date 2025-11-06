#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path

# Fix imports for the new structure
try:
    from ..core.capture_manager import CaptureManager
    from ..ml.signal_classifier import SignalClassifier
    from ..hardware.sdr_controller import SDRController
    from ..utils.config_loader import ConfigLoader
except ImportError:
    # Fallback for direct execution
    from core.capture_manager import CaptureManager
    from ml.signal_classifier import SignalClassifier
    from hardware.sdr_controller import SDRController
    from utils.config_loader import ConfigLoader

class RFCApture:
    def __init__(self):
        self.setup_logging()
        self.config = ConfigLoader()
        self.sdr = SDRController()
        self.capture_manager = CaptureManager()
        self.classifier = SignalClassifier()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        parser = argparse.ArgumentParser(
            description="Advanced RF Signal Capture and Analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  rfcapture capture --freq 433e6 --rate 2e6 --duration 10
  rfcapture classify --model cnn --input captured.iq
  rfcapture train --data dataset/ --epochs 100
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Capture command
        capture_parser = subparsers.add_parser('capture', help='Capture RF signals')
        self.setup_capture_parser(capture_parser)
        
        # Classify command
        classify_parser = subparsers.add_parser('classify', help='Classify signals')
        self.setup_classify_parser(classify_parser)
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train ML models')
        self.setup_train_parser(train_parser)
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
            
        self.execute_command(args)
    
    def setup_capture_parser(self, parser):
        parser.add_argument('--freq', type=float, required=True, 
                          help='Center frequency in Hz')
        parser.add_argument('--rate', type=float, default=2e6,
                          help='Sample rate in Hz')
        parser.add_argument('--duration', type=float, required=True,
                          help='Capture duration in seconds')
        parser.add_argument('--gain', type=float, default=20,
                          help='RF gain in dB')
        parser.add_argument('--output', '-o', type=Path, required=True,
                          help='Output file path')
        parser.add_argument('--device', type=str, default='rtlsdr',
                          help='SDR device (rtlsdr, hackrf, usrp)')
    
    def setup_classify_parser(self, parser):
        parser.add_argument('--input', '-i', type=Path, required=True,
                          help='Input IQ file or directory')
        parser.add_argument('--model', '-m', type=str, default='cnn',
                          choices=['cnn', 'lstm', 'resnet'],
                          help='Model architecture')
        parser.add_argument('--output', '-o', type=Path,
                          help='Output results file')
    
    def setup_train_parser(self, parser):
        parser.add_argument('--data', '-d', type=Path, required=True,
                          help='Training data directory')
        parser.add_argument('--model', '-m', type=str, default='cnn',
                          help='Model architecture')
        parser.add_argument('--epochs', '-e', type=int, default=100,
                          help='Number of training epochs')
        parser.add_argument('--batch-size', '-b', type=int, default=32,
                          help='Batch size')
    
    def execute_command(self, args):
        try:
            if args.command == 'capture':
                self.handle_capture(args)
            elif args.command == 'classify':
                self.handle_classify(args)
            elif args.command == 'train':
                self.handle_train(args)
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            sys.exit(1)

    def handle_capture(self, args):
        self.logger.info(f"Starting capture at {args.freq} Hz")
        self.sdr.configure(
            frequency=args.freq,
            sample_rate=args.rate,
            gain=args.gain,
            device_type=args.device
        )
        self.capture_manager.capture_to_file(
            sdr=self.sdr,
            duration=args.duration,
            output_path=args.output
        )

    def handle_classify(self, args):
        results = self.classifier.classify_signals(
            input_path=args.input,
            model_type=args.model
        )
        self.display_classification_results(results, args.output)

    def handle_train(self, args):
        self.classifier.train_model(
            data_path=args.data,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    def display_classification_results(self, results, output_path=None):
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        else:
            for result in results:
                print(f"Class: {result['class']}, Confidence: {result['confidence']:.3f}")

def main():
    app = RFCApture()
    app.run()

if __name__ == "__main__":
    main()