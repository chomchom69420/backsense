import numpy as np

start_freq = 77  # start frequency in GHz
num_antennas = 3  # number of TX antennas
d = 0.00389341 / 2  # antenna spacing in meters
theta = 45  # desired angle
lamda = (3 * 10**8) / (start_freq * 10**9)  # wavelength in meters
print(lamda)
antenna_phase = np.zeros(num_antennas)

for i in range(num_antennas):
    antenna_phase[i] = np.rad2deg((2 * np.pi / lamda) * i * d * np.sin(np.deg2rad(theta)))

antenna_phase_bits = antenna_phase / 5.625

print(antenna_phase_bits)