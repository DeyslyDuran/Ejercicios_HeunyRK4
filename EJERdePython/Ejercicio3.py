import numpy as np
import matplotlib.pyplot as plt


def dAdt(A, alpha, k, nu):
 
    
    if A <= 0: 
        return 0 
    return alpha * A * (1 - (A / k)**nu)

# Given parameters
alpha = 0.8   # Constant
k = 60        # Constant
nu = 0.25     # Constant
A0 = 1        # mm^2 (initial tumor area)
t_start = 0   # days (start time)
t_end = 30    # days (end time)
h = 0.05      # days (step size, chosen to be relatively small for accuracy)

# Create time array
t = np.arange(t_start, t_end + h, h)
num_steps = len(t)

# --- Heun's Method ---
def solve_heun(A0, t, alpha, k, nu, h):
    A_heun = np.zeros(len(t))
    A_heun[0] = A0

    for i in range(len(t) - 1):
        A_current = A_heun[i]
        
        # Predictor step (Euler's method)
        A_predictor = A_current + dAdt(A_current, alpha, k, nu) * h
        
        # Corrector step
        slope_current = dAdt(A_current, alpha, k, nu)
        slope_predictor = dAdt(A_predictor, alpha, k, nu)
        
        A_heun[i+1] = A_current + 0.5 * (slope_current + slope_predictor) * h
    return A_heun

# --- 4th-Order Runge-Kutta Method (RK4) ---
def solve_rk4(A0, t, alpha, k, nu, h):
    A_rk4 = np.zeros(len(t))
    A_rk4[0] = A0

    for i in range(len(t) - 1):
        A_current = A_rk4[i]
        
        k1 = h * dAdt(A_current, alpha, k, nu)
        k2 = h * dAdt(A_current + 0.5 * k1, alpha, k, nu)
        k3 = h * dAdt(A_current + 0.5 * k2, alpha, k, nu)
        k4 = h * dAdt(A_current + k3, alpha, k, nu)
        
        A_rk4[i+1] = A_current + (k1 + 2*k2 + 2*k3 + k4) / 6
    return A_rk4

# Solve using Heun's method
A_heun_solution = solve_heun(A0, t, alpha, k, nu, h)

# Solve using RK4 method
A_rk4_solution = solve_rk4(A0, t, alpha, k, nu, h)

# --- Print Results ---
print("--- Resultados (Método de Heun) ---")
print(f"{'Tiempo (días)':<15} | {'Área del Tumor (mm^2)':<25}")
print("-" * 45)

for i in range(0, num_steps, max(1, num_steps // 10)):
    print(f"{t[i]:<15.2f} | {A_heun_solution[i]:<25.2f}")
print(f"{t[-1]:<15.2f} | {A_heun_solution[-1]:<25.2f}")


print("\n--- Resultados (Método Runge-Kutta 4to Orden) ---")
print(f"{'Tiempo (días)':<15} | {'Área del Tumor (mm^2)':<25}")
print("-" * 45)

for i in range(0, num_steps, max(1, num_steps // 10)):
    print(f"{t[i]:<15.2f} | {A_rk4_solution[i]:<25.2f}")
print(f"{t[-1]:<15.2f} | {A_rk4_solution[-1]:<25.2f}")



plt.figure(figsize=(10, 6))
plt.plot(t, A_heun_solution, label="Método de Heun", color='blue')
plt.plot(t, A_rk4_solution, label="Método Runge-Kutta 4to Orden", color='red', linestyle='--')
plt.axhline(y=k, color='green', linestyle=':', label=f'Tamaño Límite ($k = {k}$ mm^2)')


plt.title('Crecimiento del Tumor a lo Largo del Tiempo')
plt.xlabel('Tiempo (días)')
plt.ylabel('Área del Tumor (mm$^2$)')
plt.grid(True)
plt.legend()
plt.show()