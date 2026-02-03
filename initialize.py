from config import COLUMN, EFFICIENCY
import numpy as np

def initialize_column():
    
    N = COLUMN['n_stages']
    T = np.linspace(80, 111, N)
    y = np.zeros((N, 2))
    x = np.zeros((N, 2))
    
    
    D_est = COLUMN["feed_flow"]*\
    (COLUMN["feed_z"] - 1 + COLUMN["xb_assumed"])/\
    (2*COLUMN["xb_assumed"] - 1) # over flow concn kmol/hr
    
    # always COLUMN["xb_assumed"] > 0.5
    
    B_est = COLUMN["feed_flow"] - D_est # Bottoms concn kmol/hr
    
    L_rect = COLUMN["reflux_ratio"] * D_est #rectifing section liquid flow
    V_rect = (COLUMN["reflux_ratio"] + 1) * D_est #rectifing section vapour flow
    
    L_strip = L_rect + COLUMN['feed_q'] * COLUMN["feed_flow"]
    V_strip = V_rect - (1 - COLUMN['feed_q']) * COLUMN["feed_flow"]
    
    # Assigning arrays
    L , V = np.zeros(N), np.zeros(N)



    for i in range(N):
        if i < COLUMN['feed_stage']:
            L[i] = L_rect
            V[i] = V_rect
        else:
            L[i] = L_strip
            V[i] = V_strip
    V[0] = 0 # Total condenser
    print(L_strip)
    
    #Efficiency
    E_mv = np.ones(N)*EFFICIENCY['default']
    E_mv[0] = 1
    E_mv[-1] = 1
    
    # correction for boundary streams
    """ check"""
    D_out = D_est
    B_out = COLUMN['feed_flow'] - D_out

    return x, y, T, V, L, E_mv, D_out, B_out
# COLUMN['feed_z'] = 0.4 
#