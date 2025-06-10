import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation function
def dNdt(N, k, NM):
    """
    Calculates the rate of change of population dN/dt.
    dN/dt = k * N * (NM - N)
    """
    return k * N * (NM - N)

# Given parameters
NM = 5000  # Limiting number for the population
k = 0.000095 # Constant
N0 = 100   # Initial population
t_start = 0  # Start time
t_end = 20   # End time
h = 0.1    # Step size (we'll use a relatively small step size for better accuracy)

# Create time array
t = np.arange(t_start, t_end + h, h)
num_steps = len(t)

# --- Heun's Method ---
def solve_heun(N0, t, k, NM, h):
    N_heun = np.zeros(len(t))
    N_heun[0] = N0

    for i in range(len(t) - 1):
        N_current = N_heun[i]
        
        # Predictor step (Euler's method)
        N_predictor = N_current + dNdt(N_current, k, NM) * h
        
        # Corrector step
        slope_current = dNdt(N_current, k, NM)
        slope_predictor = dNdt(N_predictor, k, NM)
        
        N_heun[i+1] = N_current + 0.5 * (slope_current + slope_predictor) * h
    return N_heun

# --- 4th-Order Runge-Kutta Method (RK4) ---
def solve_rk4(N0, t, k, NM, h):
    N_rk4 = np.zeros(len(t))
    N_rk4[0] = N0

    for i in range(len(t) - 1):
        N_current = N_rk4[i]
        
        k1 = h * dNdt(N_current, k, NM)
        k2 = h * dNdt(N_current + 0.5 * k1, k, NM)
        k3 = h * dNdt(N_current + 0.5 * k2, k, NM)
        k4 = h * dNdt(N_current + k3, k, NM)
        
        N_rk4[i+1] = N_current + (k1 + 2*k2 + 2*k3 + k4) / 6
    return N_rk4

# Solve using Heun's method
N_heun_solution = solve_heun(N0, t, k, NM, h)

# Solve using RK4 method
N_rk4_solution = solve_rk4(N0, t, k, NM, h)

# --- Print Results ---
print("--- Resultados (Heun's Method) ---")
print(f"{'Tiempo (t)':<10} | {'Población (N)':<15}")
print("-" * 30)
for i in range(0, num_steps, max(1, num_steps // 10)): # Print a subset of results for brevity
    print(f"{t[i]:<10.2f} | {N_heun_solution[i]:<15.2f}")
print(f"{t[-1]:<10.2f} | {N_heun_solution[-1]:<15.2f}")


print("\n--- Resultados (4th-Order Runge-Kutta Method) ---")
print(f"{'Tiempo (t)':<10} | {'Población (N)':<15}")
print("-" * 30)
for i in range(0, num_steps, max(1, num_steps // 10)): # Print a subset of results for brevity
    print(f"{t[i]:<10.2f} | {N_rk4_solution[i]:<15.2f}")
print(f"{t[-1]:<10.2f} | {N_rk4_solution[-1]:<15.2f}")


# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.plot(t, N_heun_solution, label="Método de Heun", color='blue')
plt.plot(t, N_rk4_solution, label="Método Runge-Kutta 4to Orden", color='red', linestyle='--')
plt.axhline(y=NM, color='green', linestyle=':', label=f'Capacidad de Carga ($N_M = {NM}$)')

plt.title('Crecimiento de la Población a lo Largo del Tiempo')
plt.xlabel('Tiempo (años)')
plt.ylabel('Población (N)')
plt.grid(True)
plt.legend()
plt.show()