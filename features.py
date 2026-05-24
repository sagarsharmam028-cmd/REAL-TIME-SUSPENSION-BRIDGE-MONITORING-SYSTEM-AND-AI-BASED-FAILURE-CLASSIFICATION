import numpy as np

def extract_features_live(buffer, sampling_rate=10.0):
    """
    Extracts advanced time-series features from a rolling sensor data buffer.
    Includes time-domain deflection statistics, a noise-robust linear regression 
    deflection slope (trend), and frequency-domain vibration statistics (FFT).
    """
    # 🔹 Extract values from buffer
    distances = [d["distance"] for d in buffer]
    weights = [d["weight"] for d in buffer]

    # 🔹 Proper vibration magnitude (physics-based)
    vibrations = [
        np.sqrt(d["ax"]**2 + d["ay"]**2 + d["az"]**2)
        for d in buffer
    ]

    # 🔹 Time-domain Deflection Features
    mean_deflection = np.mean(distances)
    max_deflection = np.max(distances)

    # 🔹 Time-domain Vibration Features
    vibration_std = np.std(vibrations)
    vibration_max = np.max(vibrations)

    # 🔹 Robust Deflection Rate (Linear Regression Slope)
    # Replaces distances[-1] - distances[0] to make trend analysis resilient to noise spikes
    N = len(distances)
    x = np.arange(N)
    # Fit y = m*x + c, where m is the slope
    slope, _ = np.polyfit(x, distances, 1)
    deflection_rate = slope

    # 🔹 Frequency-domain Vibration Features (FFT)
    # Subtracting the mean (DC component) from vibration before FFT
    vibs_detrend = vibrations - np.mean(vibrations)
    fft_magnitudes = np.abs(np.fft.rfft(vibs_detrend))
    
    if len(fft_magnitudes) > 1:
        # Ignore index 0 (residual DC component after mean subtraction)
        dominant_frequency_idx = np.argmax(fft_magnitudes[1:]) + 1
        freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
        dominant_frequency = float(freqs[dominant_frequency_idx])
        
        # Spectral energy: sum of squared magnitudes of AC frequency components
        vibration_spectral_energy = float(np.sum(fft_magnitudes[1:]**2))
    else:
        dominant_frequency = 0.0
        vibration_spectral_energy = 0.0

    return {
        # Core original features (ensures backward compatibility for old models)
        "mean_deflection": round(mean_deflection, 3),
        "max_deflection": round(max_deflection, 3),
        "vibration_std": round(vibration_std, 3),
        "vibration_max": round(vibration_max, 3),
        "deflection_rate": round(deflection_rate, 3),
        
        # Advanced new spectral features
        "dominant_frequency": round(dominant_frequency, 3),
        "vibration_spectral_energy": round(vibration_spectral_energy, 3)
    }