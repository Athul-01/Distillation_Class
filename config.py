# Base details of the model

# config.py

COMPONENTS = ["benzene", "toluene"]    # binary system

ANTOINE = {
    0: {"A": 6.90565, "B": 1211.033, "C": 220.79},
    1: {"A": 6.95334, "B": 1343.943, "C": 219.377},
}

COLUMN = {
    "pressure_atm": 1.0,
    "n_stages": 17,  #theoritical number of trays + boiler and condenser
    "feed_stage": 7,
    "reflux_ratio": 2.5,
    "feed_flow": 100.0,   # kmol/hr
    "feed_z": 0.5,        # benzene mole fraction
    "feed_q": 1.0,         # saturated liquid
    "xb_assumed": 0.95
}

EFFICIENCY = {
    "default": 0.70
}