def generate_data(n=300, mode="SAFE"):
    import numpy as np
    import pandas as pd

    time = np.arange(0, n)

    if mode == "SAFE":
        deflection = np.random.normal(3, 2.0, n)
        vibration = np.random.normal(0.6, 0.4, n)

    elif mode == "WARNING":
        deflection = np.random.normal(5, 2.5, n)
        vibration = np.random.normal(0.9, 0.5, n)

    elif mode == "DANGER":
        deflection = np.random.normal(7, 3.0, n)
        vibration = np.random.normal(1.2, 0.6, n)

    data = pd.DataFrame({
        "time": time,
        "deflection": deflection,
        "vibration": vibration
    })

    return data