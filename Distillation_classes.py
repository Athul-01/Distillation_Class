# Distillation classes

import sys
print(sys.executable)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class DistillationColumn:
    """
    DistillationColumn:
    A class to model a distilllation column using the first principles

    """
    def __init__(self, N_stages, feed_stage, efficiency, \
                 F_molar, z_F, q, RefluxRatio):

        """ 
        instance attributes:
        N_stages: number of stages in the column
        feed_stage: stage number where the feed is introduced
        efficiency: column efficiency (0 to 1)
        F_molar: molar flow rate
        z_F: feed concentration
        q : liquid feed ratio
        reflux_ratio: reflux ratio

        """

        self.N = N_stages
        self.feed_stage = feed_stage
        self.F_molar = F_molar
        self.q = q
        self.z_F = z_F
        self.RefluxRatio = RefluxRatio

        self.P_col = 760 # mmHg

        #if efficiency is given as an array or a scalar
        if np.isscalar(efficiency):
            self.E_mv = np.full(self.N, efficiency)
        else:
            self.E_mv = np.array(efficiency)

        #efficiency for reboiler(N-1) and Condenser(0) are set to 1
        self.E_mv[0] = 1.0
        self.E_mv[-1] = 1.0

        # Antoine data
        self.antoine_A = np.array([6.90565, 6.95334])
        self.antoine_B = np.array([1211.033, 1343.943])
        self.antoine_C = np.array([220.790, 219.377])

        # Variables (Storage)
        self.T = np.linspace(80.1, 110.6, self.N)  # Linear T profile guess
        self.x = np.zeros((self.N, 2)) # Liquid comp [stage, component]
        self.y = np.zeros((self.N, 2)) # Vapor comp [stage, component]
        self.L = np.zeros(self.N)      # Liquid flows
        self.V = np.zeros(self.N)      # Vapor flows

        # Initialize
        self._initialize_flows()
        self.x[:, 0] = 0.5
        self.x[:, 1] = 0.5

    def get_psat(self, T_celsius):
        """Calculate saturation pressure (mmHg) using Antoine Eq"""
        # log10(P) = A - B / (C + T)
        logP = self.antoine_A - self.antoine_B / (self.antoine_C + T_celsius)
        return 10**logP
    
    def get_K_values(self, T_celsius):
        """Calculate K-values using Raoult's Law (Ideal): K = Psat / P"""
        Psat = self.get_psat(T_celsius)
        return Psat / self.P_col
    
    def _initialize_flows(self):
        """
        Constant Molar Overflow (CMO) Assumption.
        Calculates internal flows L and V based on Reflux and Q-line.
        """
        # Overall Balance to estimate D and B
        # F = D + B
        # F*z = D*xD + B*xB  (Approximate split for flow init)
        # Assume perfect split for initialization of flows
        D_est = self.F_molar * 0.5
        B_est = self.F_molar * 0.5
        
        # Rectifying Section (Above Feed)
        L_rect = self.RefluxRatio * D_est
        V_rect = (self.RefluxRatio + 1) * D_est
        
        # Stripping Section (Below Feed)
        # L_strip = L_rect + q*F
        # V_strip = V_rect - (1-q)*F
        L_strip = L_rect + self.q * self.F_molar
        V_strip = V_rect - (1 - self.q) * self.F_molar
        
        # Assign to arrays
        for i in range(self.N):
            if i < self.feed_stage:
                self.L[i] = L_rect
                self.V[i] = V_rect
            else:
                self.L[i] = L_strip
                self.V[i] = V_strip

        # print(L_rect)
        
        # Correction for boundary streams
        self.D = D_est # Store D
        self.B = self.F_molar - self.D
        
        # V[0] is physically 0 (Total Condenser liquid out), but mathematically
        # the balance is easier if we track internal traffic. 
        # L[0] = Reflux. Distillate is separate.
        self.L[0] = L_rect
        self.V[0] = 0 

    def thomas_algorithm(self, a, b, c, d):
        """
        Tridiagonal Matrix Solver (TDMA).
        Solves Ax = d for x.
        """
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        x_sol = np.zeros(n)

        # Forward
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        for i in range(1, n):
            if i < n-1:
                c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i-1])
            denom = (b[i] - a[i] * c_prime[i-1])
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        # Backward
        x_sol[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x_sol[i] = d_prime[i] - c_prime[i] * x_sol[i+1]
            
        return x_sol
    
    def solve_mass_balance_component(self, comp_idx):
        """
        Builds and solves the tridiagonal matrix for component 'comp_idx'.
        Includes Murphree Efficiency logic: y_n = E * K * x_n + (1-E) * y_n+1_old
        """
        N = self.N
        A = np.zeros(N) #Lower
        B = np.zeros(N) # Main
        C = np.zeros(N) # Upper
        RHS = np.zeros(N) # Right hand Side

        # Get thermo data
        K = np.array([self.get_K_values(t)[comp_idx] for t in self.T])

        # Use previous iteration's y for the efficiency lag term
        # y_lag = (1- E) * y_old
        y_old = self.y[:, comp_idx]

        for j in range(N):
            # Flows
            # L_above (L_{j-1})
            L_in = self.L[j-1] if j > 0 else 0

            # V_below (V_{j+1})
            V_in = self.V[j+1] if j < N-1 else 0

            L_out = self.L[j]
            V_out = self.V[j]

            y_incoming_lag = y_old[j+1] if j < N-1 else 0

            # Component entering from liquid above: L_{j-1} * x_{j-1}
            if j > 0:
                A[j] = L_in

            # Component leaving this stage (Liquid + Vapor)
            B[j] = - (L_out + V_out * self.E_mv[j] * K[j])

            # Component entering from vapor below: V_{j+1} * y_{j+1}
            if j < N-1:
                C[j] = V_in * self.E_mv[j+1] * K[j+1]

            # RHS / Constants
            rhs_val = 0.0
            
            # Add Feed
            if j == self.feed_stage:
                rhs_val -= self.F_molar * self.z_F[comp_idx]

            term_loss_lag = V_out * (1 - self.E_mv[j]) * y_incoming_lag
            rhs_val += term_loss_lag

            # Term from y_{j+1} entering from below:
            if j < N-1:
                y_incoming_from_2_below = y_old[j+2] if j < N-2 else 0
                term_gain_lag = V_in * (1 - self.E_mv[j+1]) * y_incoming_from_2_below
                rhs_val -= term_gain_lag # moved to RHS
            
            RHS[j] = rhs_val

        # Special Case: Total Condenser (Stage 0)
        B[0] = - (self.L[0] + self.D)
        # Special Case: Reboiler (Stage N-1)
        B[N-1] = - (self.B + self.V[N-1] * self.E_mv[N-1] * K[N-1])
        return self.thomas_algorithm(A, B, C, RHS)
    
    def solve(self, max_iter=100, tol = 1e-4):
        """ Main iteration loop"""
        error = 1.0
        iter_count = 0
        
        while error > tol and iter_count < max_iter:
            iter_count += 1
            # print(iter_count)
            x_new = np.zeros_like(self.x)
            
            # Solve Mass Balances for x (Benzene & Toluene)
            x_benz = self.solve_mass_balance_component(0)
            x_tol = self.solve_mass_balance_component(1)
            
            # Normalize Compositions (Summation Eq)
            sum_x = x_benz + x_tol
            x_benz = x_benz / sum_x
            x_tol = x_tol / sum_x

            # print('x_benz', x_benz)
            # print('x_tol', x_tol)
            
            x_new[:, 0] = x_benz

            x_new[:, 1] = x_tol
            
            # Update Temperature (Bubble Point)
            # Find T such that sum(K_i * x_i) = 1
            T_new = np.zeros(self.N)
            for j in range(self.N):
                def bubble_func(T_guess):
                    K = self.get_K_values(T_guess)
                    return K[0]*x_new[j,0] + K[1]*x_new[j,1] - 1.0
                
                # Use fsolve to find bubble point
                # Use current T[j] as initial guess
                sol = fsolve(bubble_func, self.T[j])
                T_new[j] = sol[0]
            
            
            # Update Vapor Compositions (y)
            
            for j in range(self.N):
                K_vals = self.get_K_values(T_new[j])
                # Ideal Equilibrium
                y_star_benz = K_vals[0] * x_new[j, 0]
                y_star_tol  = K_vals[1] * x_new[j, 1]
                
                # Apply Efficiency: y = y_in + E(y* - y_in)
                # y_in is vapor from stage below (j+1)
                if j < self.N - 1:
                    y_in_benz = self.y[j+1, 0] # Use previous iter value for stability
                    y_in_tol  = self.y[j+1, 1]
                else:
                    y_in_benz = 0 # No vapor into reboiler
                    y_in_tol = 0
                
                self.y[j, 0] = y_in_benz + self.E_mv[j] * (y_star_benz - y_in_benz)
                self.y[j, 1] = y_in_tol  + self.E_mv[j] * (y_star_tol - y_in_tol)

            # Check Convergence (Temperature Change)
            error = np.max(np.abs(T_new - self.T))
            
            # Update
            self.T = 0.8 * self.T + 0.2 * T_new # Damping
            self.x = x_new
            
        return iter_count, error
    

# col = DistillationColumn(N_stages=17, feed_stage= 8, efficiency=0.7, \
#                          F_molar=100, z_F=np.array([0.5, 0.5]), q = 1, \
#                             RefluxRatio= 2.5)
# col.solve()
# print(col.x)

def run_scenario_A():
    """Scenario A: Equipment Deterioration"""
    print("\n--- Running Scenario A: Efficiency Degradation ---")
    
    # Define Cases
    cases = {
        "Baseline (70%)": 0.70,
        "Degraded (60%)": 0.60,
        "Localized (Trays 5-8 at 50%)": "localized",
        "Progressive (70% -> 55%)": "progressive"
    }
    
    results = {}
    
    for name, eff_setting in cases.items():
        efficiency = np.full(17, 0.70) # Default
        
        if eff_setting == "localized":
            # Stages 0..16. Feed at 8. Trays are usually counted from top.
            # Let's assume indices 5,6,7,8 are damaged.
            efficiency[5:9] = 0.50
        elif eff_setting == "progressive":
            # Linear decrease from 0.70 at top (idx 1) to 0.55 at bottom (idx 15)
            # Exclude condenser/reboiler from degradation usually, but let's apply to trays
            linspace = np.linspace(0.70, 0.55, 15)
            efficiency[1:16] = linspace
        elif isinstance(eff_setting, float):
            efficiency = np.full(17, eff_setting)
            
        # Reset Condenser/Reboiler to 1.0 (Theoretical stages)
        efficiency[0] = 1.0
        efficiency[-1] = 1.0
        
        # Run Model
        # N=17 (15 trays + 1 Cond + 1 Reb), Feed @ 8 (Stage 8 is 9th physical stage approx)
        col = DistillationColumn(N_stages=17, feed_stage= 8, efficiency=efficiency, \
                          F_molar=100, z_F=np.array([0.5, 0.5]), q = 1, \
                            RefluxRatio= 2.5)
        col.solve()
        
        # Store Distillate Benzene Purity
        results[name] = col.x[0, 0] # Stage 0, Benzene
        
        # Plot Profile for this case
        plt.plot(col.x[:, 0], range(17), label=f"{name} (xD={col.x[0,0]:.3f})")

    plt.gca().invert_yaxis()
    plt.title("Scenario A: Liquid Composition Profiles (Benzene)")
    plt.ylabel("Stage Number")
    plt.xlabel("Benzene Mole Fraction (x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Results (Benzene Purity):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

a = run_scenario_A()









