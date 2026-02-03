# from config import ANTOINE, COLUMN, COMPONENTS, EFFICIENCY
from initialize import initialize_column
from solution import solve
from post_processing import save_as_dataframe, save_as_pickle, save_as_text
def run_simulation():
    print('Simulation has begun')
    x, y, T, V, L, E_mv, D_out, B_out = initialize_column()
    test = solve(x, y, T, V, L, E_mv, D_out, B_out)
    x_final, y_final, T_final, ops, iters, error = test
    if error < 1e-3:
        print(f"\n Simulation Converged in {iters} iterations.")
        print(f"Max Temperature Error: {error:.2e}")
        
        # Display Key Performance Indicators
        print("\n--- Key Results ---")
        print(f"Distillate Purity (Benzene): {x_final[0, 0]:.4f}")
        print(f"Bottoms Purity (Toluene):    {x_final[-1, 1]:.4f}")
        print(f"Top Temperature:             {T_final[0]:.2f} °C")
        print(f"Bottom Temperature:          {T_final[-1]:.2f} °C")
    # save_as_dataframe(test)
    # save_as_pickle(test)
    # save_as_text(test)
if __name__ == "__main__":
    run_simulation()

