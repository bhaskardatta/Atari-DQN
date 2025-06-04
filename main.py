#!/usr/bin/env python3
# filepath: /Users/bhaskar/Desktop/atari/main.py
"""
Main entry point for the adaptive RL project.
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from examples.train_agent import main as train_main
from examples.evaluate_agent import main as evaluate_main
from examples.analyze_results import main as analyze_main


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive RL for Atari Breakout',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --config configs/default.yaml
  python main.py evaluate --checkpoint results/models/best_model.pth --episodes 10
  python main.py analyze --results-dir results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                              help='Path to config file')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Path to checkpoint to resume from')
    train_parser.add_argument('--device', type=str, default='auto',
                              help='Device to use (cpu, cuda, auto)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained agent')
    eval_parser.add_argument('--config', type=str, default='configs/default.yaml',
                             help='Path to config file')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to trained model checkpoint')
    eval_parser.add_argument('--episodes', type=int, default=10,
                             help='Number of episodes to evaluate')
    eval_parser.add_argument('--record-video', action='store_true',
                             help='Record video of first episode')
    eval_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use (cpu, cuda, auto)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze training results')
    analyze_parser.add_argument('--results-dir', type=str, required=True,
                                help='Path to results directory')
    analyze_parser.add_argument('--output-dir', type=str, default=None,
                                help='Output directory for plots')
    analyze_parser.add_argument('--format', type=str, default='png',
                                choices=['png', 'pdf', 'svg'],
                                help='Plot format')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Modify sys.argv for train_main
        sys.argv = ['train_agent.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        if args.device:
            sys.argv.extend(['--device', args.device])
        train_main()
        
    elif args.command == 'evaluate':
        # Modify sys.argv for evaluate_main
        sys.argv = ['evaluate_agent.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        sys.argv.extend(['--checkpoint', args.checkpoint])
        sys.argv.extend(['--episodes', str(args.episodes)])
        if args.record_video:
            sys.argv.append('--record-video')
        if args.device:
            sys.argv.extend(['--device', args.device])
        evaluate_main()
        
    elif args.command == 'analyze':
        # Modify sys.argv for analyze_main
        sys.argv = ['analyze_results.py']
        sys.argv.extend(['--results-dir', args.results_dir])
        if args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        sys.argv.extend(['--format', args.format])
        analyze_main()
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
