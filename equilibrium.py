import numpy as np
from config import ANTOINE, COLUMN
from pressure import sat_p
from typing import List, Dict

def k_value(T: float, comp: Dict):
    # calculating K values using raoult's law
    K = sat_p(T, comp)/COLUMN["pressure_atm"]
    return K