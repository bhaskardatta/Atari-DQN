#!/usr/bin/env python3
# filepath: /Users/bhaskar/Desktop/atari/examples/analyze_results.py
"""
Example script for analyzing training results and generating plots.
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import json

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logger, get_experiment_logger
from src.analysis.performance import PerformanceAnalyzer
from src.analysis.visualization import (
    GameplayVisualizer, 
    EnvironmentVisualizer,
    save_visualization
)


def load_training_data(results_dir):
    """
    Load training data from results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary with loaded data
    """
    results_path = Path(results_dir)
    data = {}
    
    # Load metrics CSV
    metrics_file = results_path / 'logs' / 'metrics.csv'
    if metrics_file.exists():
        data['metrics'] = pd.read_csv(metrics_file)
    
    # Load adaptation events
    adaptation_file = results_path / 'logs' / 'adaptation_events.json'
    if adaptation_file.exists():
        with open(adaptation_file, 'r') as f:
            data['adaptation_events'] = json.load(f)
    
    # Load difficulty history
    difficulty_file = results_path / 'logs' / 'difficulty_history.json'
    if difficulty_file.exists():
        with open(difficulty_file, 'r') as f:
            data['difficulty_history'] = json.load(f)
    
    # Load curriculum history
    curriculum_file = results_path / 'logs' / 'curriculum_history.json'
    if curriculum_file.exists():
        with open(curriculum_file, 'r') as f:
            data['curriculum_history'] = json.load(f)
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: results_dir/plots)')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Plot format')
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(name="analysis", level='INFO')
    logger = get_experiment_logger("analysis")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.results_dir) / 'plots')
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading training data from {args.results_dir}")
    
    # Load training data
    try:
        training_data = load_training_data(args.results_dir)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return
    
    if not training_data:
        logger.error("No training data found")
        return
    
    logger.info(f"Loaded data with keys: {list(training_data.keys())}")
    
    # Initialize analyzers
    performance_analyzer = PerformanceAnalyzer(save_plots=True, plot_format=args.format)
    gameplay_visualizer = GameplayVisualizer()
    env_visualizer = EnvironmentVisualizer()
    
    # Generate performance analysis
    if 'metrics' in training_data:
        logger.info("Generating performance analysis plots...")
        
        # Convert DataFrame to dictionary format expected by analyzer
        metrics_dict = {
            'episodes': training_data['metrics']['episode'].tolist(),
            'scores': training_data['metrics']['score'].tolist(),
            'losses': training_data['metrics'].get('loss', []).tolist() if 'loss' in training_data['metrics'].columns else [],
            'exploration_rates': training_data['metrics'].get('epsilon', []).tolist() if 'epsilon' in training_data['metrics'].columns else []
        }
        
        analysis_results = performance_analyzer.analyze_training(
            metrics_dict, 
            save_path=str(output_path)
        )
        
        # Save analysis summary
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
    
    # Generate adaptation analysis
    if 'adaptation_events' in training_data:
        logger.info("Generating adaptation timeline...")
        
        fig = gameplay_visualizer.create_adaptation_timeline(
            training_data['adaptation_events']
        )
        save_visualization(
            fig, 
            str(output_path / f'adaptation_timeline.{args.format}'),
            format=args.format
        )
        plt.close(fig)
    
    # Generate difficulty progression plots
    if 'difficulty_history' in training_data:
        logger.info("Generating difficulty progression plots...")
        
        fig = env_visualizer.create_difficulty_progression(
            training_data['difficulty_history']
        )
        save_visualization(
            fig,
            str(output_path / f'difficulty_progression.{args.format}'),
            format=args.format
        )
        plt.close(fig)
    
    # Generate exploration visualization
    if 'metrics' in training_data and 'epsilon' in training_data['metrics'].columns:
        logger.info("Generating exploration analysis...")
        
        exploration_rates = training_data['metrics']['epsilon'].tolist()
        fig = gameplay_visualizer.create_exploration_visualization(exploration_rates)
        save_visualization(
            fig,
            str(output_path / f'exploration_analysis.{args.format}'),
            format=args.format
        )
        plt.close(fig)
    
    # Generate performance comparison by curriculum stage
    if 'curriculum_history' in training_data and 'metrics' in training_data:
        logger.info("Generating curriculum performance analysis...")
        
        # Group performance by curriculum stage
        performance_by_stage = {}
        curriculum_data = training_data['curriculum_history']
        metrics_data = training_data['metrics']
        
        for stage_info in curriculum_data:
            stage_name = stage_info.get('stage', 'unknown')
            start_episode = stage_info.get('start_episode', 0)
            end_episode = stage_info.get('end_episode', len(metrics_data))
            
            # Get scores for this stage
            stage_mask = (metrics_data['episode'] >= start_episode) & (metrics_data['episode'] < end_episode)
            stage_scores = metrics_data[stage_mask]['score'].tolist()
            
            if stage_scores:
                performance_by_stage[stage_name] = stage_scores
        
        if performance_by_stage:
            fig = gameplay_visualizer.create_performance_comparison(performance_by_stage)
            save_visualization(
                fig,
                str(output_path / f'curriculum_performance.{args.format}'),
                format=args.format
            )
            plt.close(fig)
    
    logger.info(f"Analysis complete! Plots saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING ANALYSIS SUMMARY")
    print("="*60)
    
    if 'metrics' in training_data:
        metrics = training_data['metrics']
        print(f"Total Episodes: {len(metrics)}")
        print(f"Final Score: {metrics['score'].iloc[-1]:.2f}")
        print(f"Best Score: {metrics['score'].max():.2f}")
        print(f"Average Score (last 100): {metrics['score'].tail(100).mean():.2f}")
    
    if 'adaptation_events' in training_data:
        events = training_data['adaptation_events']
        print(f"Adaptation Events: {len(events)}")
        
        event_types = {}
        for event in events:
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count}")
    
    if 'curriculum_history' in training_data:
        stages = training_data['curriculum_history']
        print(f"Curriculum Stages: {len(stages)}")
        for stage in stages:
            print(f"  {stage.get('stage', 'unknown')}: Episodes {stage.get('start_episode', 0)}-{stage.get('end_episode', 0)}")
    
    print(f"\nPlots saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
