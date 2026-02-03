import numpy as np
from config import ANTOINE
from typing import List, Dict

def sat_p(T: float, comp: Dict):
    #calculating saturation pressure using antione eqn.
    A, B, C = ANTOINE[comp].values()
    p_mmHg = 10**(A-B/(C + T))
    return p_mmHg/760