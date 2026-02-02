"""
holzbot-roof: roof analysis from wall mask images.
"""

from roof_calc.algorithm import calculate_roof_from_walls, select_algorithm
from roof_calc.visualize import *  # re-export visualize helpers
from roof_calc.masks import *  # re-export mask generators

__all__ = [
    "calculate_roof_from_walls",
    "select_algorithm",
]

