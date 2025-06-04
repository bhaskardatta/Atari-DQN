# Copilot Instructions

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

This is an adaptive reinforcement learning project for Atari Breakout with dynamic difficulty. The project implements:

- Deep Q-Network (DQN) from scratch with experience replay
- Dynamic Breakout environment with variable difficulty
- Curriculum learning and adaptation mechanisms
- Performance analysis and visualization tools

Key components:
- `src/dqn/`: Core DQN implementation
- `src/environment/`: Modified Breakout environment with dynamic difficulty
- `src/training/`: Training loops and curriculum learning
- `src/analysis/`: Performance analysis and visualization tools

When generating code, please:
- Use PyTorch for neural networks
- Follow object-oriented design patterns
- Include proper error handling and logging
- Add comprehensive docstrings and type hints
- Use numpy for numerical computations
- Consider GPU acceleration where applicable
