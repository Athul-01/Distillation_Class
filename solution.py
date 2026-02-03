import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from config import ANTOINE, COLUMN, EFFICIENCY
from mass_balance import solve_mass_balance
from equilibrium import k_value

def solve(x, y, T, V, L, E_mv, D_out, B_out, max_iter = 100, tol = 1e-4):
    N = COLUMN['n_stages']
    error = 1
    iter_count = 0
    while error > tol and iter_count < max_iter:
        
        
        iter_count +=1
        # print(iter_count)
        x_new = np.zeros_like(x)

        # solve mass balance
        x_benz = solve_mass_balance(0, x, y, T, V, L, E_mv, D_out, B_out)
        x_tol = solve_mass_balance(1, x, y, T, V, L, E_mv, D_out, B_out)
        # print('x_benz', x_benz)
        # print('x_tol', x_tol)

        # Normalize Compositions (Summation Eq)
        sum_x = x_benz + x_tol
        x_benz = x_benz / sum_x
        x_tol = x_tol / sum_x

        x_new[:, 0] = x_benz
        x_new[:, 1] = x_tol

        # Update Temperature (Bubble Point)
        T_new = np.zeros(N)
        for j in range(N):
            def bubble_func(T_guess):
                K_benz = k_value(T_guess, 0)
                K_tol = k_value(T_guess, 1)
                
                return K_benz*x_new[j, 0] + K_tol*x_new[j,1]-1

            sol = fsolve(bubble_func, T[j])
            T_new[j] = sol[0]
        # print(T_new)
        # Update Vapor Compositions (y)
        for j in range(N):
            K_vals_benz = k_value(T_new[j], 0)
            K_vals_tol = k_value(T_new[j], 1)
            y_star_benz = K_vals_benz * x_new[j, 0]
            y_star_tol  = K_vals_tol * x_new[j, 1]

            if j < N-1:
                y_in_benz = y[j+1, 0]
                y_in_tol = y[j+1, 1]

            else:
                y_in_benz = 0
                y_in_tol = 0

            y[j,0] = y_in_benz + E_mv[j] * (y_star_benz - y_in_benz)
            y[j,1] = y_in_tol + E_mv[j] * (y_star_tol - y_in_tol)

        # check convergence
        error = np.max(np.abs(T_new - T))

        T = 0.8* T + 0.2* T_new
        x = x_new
        
    operating_var = {"pressure_atm": COLUMN['pressure_atm'],
        "n_stages": COLUMN['n_stages'],
        "feed_stage": COLUMN['feed_stage'],
        "reflux_ratio": COLUMN['reflux_ratio'],
        "feed_flow": COLUMN['feed_flow'],   # kmol/hr
        "feed_z": COLUMN['feed_z'],        # benzene mole fraction
        "feed_q": COLUMN['feed_q'],
        "efficiency" : EFFICIENCY['default']}

    return x, y, T, operating_var, iter_count, error