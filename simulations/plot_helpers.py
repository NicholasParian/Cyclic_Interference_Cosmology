
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_space(y, title, filename):
    phi, phidot = y
    plt.figure(figsize=(6,6))
    plt.plot(phi, phidot, lw=1.5)
    plt.xlabel('$\phi$')
    plt.ylabel('$\dot{\phi}$')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_energy(t, energy, filename):
    plt.figure()
    plt.plot(t, energy, lw=1.5)
    plt.axhline(1.0, color='r', linestyle='--', label='Critical Density $\rho_c$')
    plt.xlabel('Time')
    plt.ylabel('Total Energy Density')
    plt.title('Total Energy Density Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_3d_ensemble(filename):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _ in range(50):
        phi = np.random.normal(0, 1, 100)
        phidot = np.random.normal(0, 0.5, 100)
        chi = np.random.normal(0, 1, 100)
        ax.plot(phi, phidot, chi, alpha=0.3)
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\dot{\phi}$')
    ax.set_zlabel('$\chi$')
    ax.set_title('Ensemble Attractor Cloud')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
