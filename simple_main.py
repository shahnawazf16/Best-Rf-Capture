#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path

class SimpleRFCapture:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def run(self):
        parser = argparse.ArgumentParser(
            description="Simple RF Signal Capture and Analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  rfcapture capture --freq 433e6 --duration 10 --output test.npy
  rfcapture train --model rf
  rfcapture classify --input test.npy --model rf
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Capture command
        capture_parser = subparsers.add_parser('capture', help='Capture RF signals')
        capture_parser.add_argument('--freq', type=float, default=433e6, 
                                  help='Center frequency in Hz')
        capture_parser.add_argument('--duration', type=float, default=5,
                                  help='Capture duration in seconds')
        capture_parser.add_argument('--output', '-o', type=Path, required=True,
                                  help='Output file path')
        
        # Classify command
        classify_parser = subparsers.add_parser('classify', help='Classify signals')
        classify_parser.add_argument('--input', '-i', type=Path, required=True,
                                   help='Input IQ file')
        classify_parser.add_argument('--model', '-m', type=str, default='rf',
                                   choices=['rf', 'svm'], help='Model type')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train ML models')
        train_parser.add_argument('--model', '-m', type=str, default='rf',
                                choices=['rf', 'svm'], help='Model type')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
            
        self.execute_command(args)
    
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
        from src.hardware.sdr_controller import SDRController
        from src.core.capture_manager import CaptureManager
        
        self.logger.info(f"Starting capture at {args.freq} Hz")
        sdr = SDRController()
        cm = CaptureManager()
        
        sdr.configure(
            frequency=args.freq,
            sample_rate=2e6,
            gain=20,
            device_type='rtlsdr'
        )
        cm.capture_to_file(
            sdr=sdr,
            duration=args.duration,
            output_path=args.output
        )

    def handle_classify(self, args):
        from src.ml.light_classifier import LightSignalClassifier
        
        self.logger.info(f"Classifying signals from {args.input}")
        classifier = LightSignalClassifier()
        results = classifier.classify_signals(
            input_path=args.input,
            model_type=args.model
        )
        
        for result in results[:5]:  # Show first 5 results
            print(f"Segment {result['segment']}: {result['class']} (confidence: {result['confidence']:.3f})")

    def handle_train(self, args):
        from src.ml.light_classifier import LightSignalClassifier
        
        self.logger.info(f"Training {args.model} model")
        classifier = LightSignalClassifier()
        results = classifier.train_model(
            data_path=Path('.'),
            model_type=args.model
        )
        
        print(f"Model trained with accuracy: {results['accuracy']:.3f}")

def main():
    app = SimpleRFCapture()
    app.run()

if __name__ == "__main__":
    main()