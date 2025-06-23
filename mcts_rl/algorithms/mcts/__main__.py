"""The main training script to train RLHF using PPO algorithm."""

import sys

## CH_adapted
import debugpy
debugpy.listen(("0.0.0.0", 5678))  # listens on port 5678 for VSCode debugger
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached, continuing...")

from mcts_rl.algorithms.mcts.main import main


if __name__ == '__main__':
    sys.exit(main())
