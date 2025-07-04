"""The main training script to train RLHF using PPO algorithm."""

import sys
import os

## CH_adapted
#if int(os.environ.get("RANK", 0)) == 0:  # or use "LOCAL_RANK"
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached, continuing...")

from mcts_rl.algorithms.mcts.main import main


if __name__ == '__main__':
    sys.exit(main())
