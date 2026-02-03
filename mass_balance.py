from config import ANTOINE, COLUMN, EFFICIENCY
import numpy as np
from equilibrium import k_value
from Algorithm import thomas_algorithm

def solve_mass_balance(comp, x, y, T, V, L, E_mv, D_out, B_out):
    N = COLUMN['n_stages']
    z_f = np.array([COLUMN['feed_z'], 1-COLUMN['feed_z']])
    q = COLUMN['feed_q']
    
 
    A = np.zeros(N) # lower diagonal
    B = np.zeros(N) # main diagonal
    C = np.zeros(N) # upper diagonal
    RHS = np.zeros(N)

    # getting the K values
    K = np.array([k_value(t, comp) for t in T])
    # print(K)
    # x = np.clip(x, 1e-6, 1-1e-6)
    x[:] = 0.5
    y_old = y[:, comp]

    for j in range(N):
        L_in = L[j-1] if j> 0 else 0
        # print('L_in', L_in)
        V_in = V[j+1] if j < N-1 else 0

        L_out = L[j]
        V_out = V[j]

        y_incoming = y_old[j+1] if j < N-1 else 0

        # matrix coeffs
        # 1. Component entering from liquid above
        if j > 0:
            A[j] = L_in
        # 2. Component leaving this stage (Liquid + Vapor)
        B[j] = - (L_out + V_out * E_mv[j]*K[j])
        # 3. Component entering from vapor below

        if j< N-1:
            C[j] = V_in * E_mv[j+1] * K[j+1]

        # 4. RHS / Constants
        rhs_val = 0

        if j == COLUMN['feed_stage']:
            z1 = COLUMN['feed_z'] if comp == 0 else (1 -COLUMN['feed_z'])# comp index check 
            rhs_val -= COLUMN['feed_flow'] * z1
            
        term_loss_lag = V_out * (1-E_mv[j]) * y_incoming
        rhs_val += term_loss_lag
        if j < N-1:
            y_incoming_from_2_below = y_old[j+2] if j< N-2 else 0
            term_gain_lag = V_in*(1-E_mv[j+1])*y_incoming_from_2_below
            rhs_val -= term_gain_lag

        RHS[j] = rhs_val

    # Special Case: Total Condenser (Stage 0)
    B[0] = -(L[0] + D_out)
    # Special Case: Reboiler (Stage N-1)
    B[N-1] = - (B_out + V[N-1]* E_mv[N-1] * K[N-1])

    # print('A', A)
    # print('B', B)
    # print('C', C)

    return thomas_algorithm(A, B, C, RHS)