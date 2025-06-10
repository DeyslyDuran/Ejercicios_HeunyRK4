import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation function
def dvdt(v, m, g, k):
    """
    Calculates the rate of change of velocity dv/dt based on the given equation.
    dv/dt = -m*g + k*v^2
    """
    return -m * g + k * (v**2)

# Given parameters
m = 5      # kg (mass of the object)
g = 9.81   # m/s^2 (acceleration due to gravity)
k = 0.05   # kg/m (constant for resistance)
v0 = 0     # m/s (initial velocity)
t_start = 0  # s (start time)
t_end = 15   # s (end time)
h = 0.01   # s (step size, adjusted for potentially rapid changes in velocity)

# Create time array
t = np.arange(t_start, t_end + h, h)
num_steps = len(t)

# --- Heun's Method ---
def solve_heun(v0, t, m, g, k, h):
    v_heun = np.zeros(len(t))
    v_heun[0] = v0

    for i in range(len(t) - 1):
        v_current = v_heun[i]
        
        # Predictor step (Euler's method)
        v_predictor = v_current + dvdt(v_current, m, g, k) * h
        
        # Corrector step
        slope_current = dvdt(v_current, m, g, k)
        slope_predictor = dvdt(v_predictor, m, g, k)
        
        v_heun[i+1] = v_current + 0.5 * (slope_current + slope_predictor) * h
    return v_heun

# --- 4th-Order Runge-Kutta Method (RK4) ---
def solve_rk4(v0, t, m, g, k, h):
    v_rk4 = np.zeros(len(t))
    v_rk4[0] = v0

    for i in range(len(t) - 1):
        v_current = v_rk4[i]
        
        k1 = h * dvdt(v_current, m, g, k)
        k2 = h * dvdt(v_current + 0.5 * k1, m, g, k)
        k3 = h * dvdt(v_current + 0.5 * k2, m, g, k)
        k4 = h * dvdt(v_current + k3, m, g, k)
        
        v_rk4[i+1] = v_current + (k1 + 2*k2 + 2*k3 + k4) / 6
    return v_rk4

# Solve using Heun's method
v_heun_solution = solve_heun(v0, t, m, g, k, h)

# Solve using RK4 method
v_rk4_solution = solve_rk4(v0, t, m, g, k, h)

# --- Print Results ---
print("--- Resultados (Método de Heun) ---")
print(f"{'Tiempo (t)':<10} | {'Velocidad (v)':<15}")
print("-" * 30)
# Print a subset of results for brevity, e.g., every 100th step
for i in range(0, num_steps, max(1, num_steps // 10)):
    print(f"{t[i]:<10.2f} | {v_heun_solution[i]:<15.2f}")
print(f"{t[-1]:<10.2f} | {v_heun_solution[-1]:<15.2f}")


print("\n--- Resultados (Método Runge-Kutta 4to Orden) ---")
print(f"{'Tiempo (t)':<10} | {'Velocidad (v)':<15}")
print("-" * 30)
# Print a subset of results for brevity, e.g., every 100th step
for i in range(0, num_steps, max(1, num_steps // 10)):
    print(f"{t[i]:<10.2f} | {v_rk4_solution[i]:<15.2f}")
print(f"{t[-1]:<10.2f} | {v_rk4_solution[-1]:<15.2f}")


# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.plot(t, v_heun_solution, label="Método de Heun", color='blue')
plt.plot(t, v_rk4_solution, label="Método Runge-Kutta 4to Orden", color='red', linestyle='--')

plt.title('Velocidad de un Objeto en Caída Libre a lo Largo del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.grid(True)
plt.legend()
plt.show()