import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def save_as_pickle(test):
    with open('simulation_results.pkl', 'wb') as file:
        pickle.dump(test, file)
    print('file saved successfully')
def save_as_text(test):
    
    with open('simulation_results.txt', 'w') as f:
        f.write(", ".join(map(str, test)))
    print('file saved successfully')
def save_as_dataframe(test):
    x_val = test[0]
    y_val = test[1]
    T_val = test[2]
    components=("benzene", "toluene")
    N = len(T_val)

    df = pd.DataFrame({
        "stage": np.arange(N),
        "T": T_val,
        f"x_{components[0]}": x_val[:, 0],
        f"x_{components[1]}": x_val[:, 1],
        f"y_{components[0]}": y_val[:, 0],
        f"y_{components[1]}": y_val[:, 1],
    })
    df.to_csv('simulation_results.csv', index=False) 
