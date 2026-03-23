"""Backward-compatible aggregate module.

The implementation is split across dedicated modules.
"""

from .core.config import *  # noqa: F401,F403
from .rl.policy import *  # noqa: F401,F403
from .core.entities import *  # noqa: F401,F403
from .env.world import *  # noqa: F401,F403
from .viz.renderer import *  # noqa: F401,F403
from .train.runner import run  # noqa: F401


if __name__ == "__main__":
    run()
