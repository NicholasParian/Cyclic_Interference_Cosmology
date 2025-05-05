
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from simulations.plot_helpers import plot_phase_space, plot_energy, plot_3d_ensemble

def bounce_condition(rho, rho_c=1.0):
    return rho >= rho_c

def friedmann_eq(rho, rho_c=1.0):
    return np.sqrt((8 * np.pi / 3) * rho * (1 - rho / rho_c))

def scalar_field_dynamics(t, y, potential, dV_dphi, rho_c=1.0):
    phi, phidot = y
    rho = 0.5 * phidot**2 + potential(phi)
    H = friedmann_eq(rho, rho_c)
    dphidt = phidot
    dphidotdt = -3 * H * phidot - dV_dphi(phi)
    return [dphidt, dphidotdt]

def quadratic_potential(phi, m=1.0):
    return 0.5 * m**2 * phi**2

def dV_quadratic(phi, m=1.0):
    return m**2 * phi

def periodic_potential(phi, Lambda=1.0, f=1.0):
    return Lambda**4 * (1 - np.cos(phi / f))

def dV_periodic(phi, Lambda=1.0, f=1.0):
    return Lambda**4 * np.sin(phi / f) / f

def simulate(phi0, phidot0, potential_fn, dV_fn, t_max=10, dt=0.01):
    y0 = [phi0, phidot0]
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(
        fun=lambda t, y: scalar_field_dynamics(t, y, potential_fn, dV_fn),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol.t, sol.y

def run_all_simulations(output_dir):
    # Quadratic attractor
    t1, y1 = simulate(2.0, 0.0, quadratic_potential, dV_quadratic)
    plot_phase_space(y1, title='Quadratic Potential Attractor', filename=f"{output_dir}/quadratic_attractor.png")

    # Periodic attractor
    t2, y2 = simulate(3.0, 0.0, periodic_potential, dV_periodic)
    plot_phase_space(y2, title='Periodic Potential Attractor', filename=f"{output_dir}/periodic_attractor.png")

    # Energy tracking
    energy = 0.5 * y1[1]**2 + quadratic_potential(y1[0])
    plot_energy(t1, energy, filename=f"{output_dir}/total_energy_density.png")

    # Ensemble attractor cloud (3D plot with noise)
    plot_3d_ensemble(filename=f"{output_dir}/ensemble_attractor_cloud.png")
