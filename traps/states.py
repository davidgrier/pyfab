"""states.py: possible states for optical traps"""

from enum import Enum


class states(Enum):
    static = 0
    normal = 1
    selected = 2
    grouping = 3
    inactive = 4
