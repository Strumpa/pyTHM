### Python3 Script aimed at comparing pyTHM results for different power distribution shapes.
# It was noticed that THM: donjon5 ev 3785 yields significantly different results for flat vs sinusoidal power distribution shapes.
# The purpose of this script is to investigate this difference in pyTHM in order to get insight as to where THM: D5 might fail.

# Author : R. Guasch
# Date : 25/05/2025
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pyTHM.Solver import pyTHM_solver
import pandas as pd
import os

def guess_power_density_sine(Ptot, h, r, num_control_volumes): # New version, supposed to be more accurate but introduced alot of instability in the TH solution
    """
    Compute the power density in a cylinder with a sine-shaped profile along the axial direction.

    Parameters:
    Ptot (float): Total power released in the cylinder [W].
    h (float): Total height of the cylinder [m].
    r (float): Radius of the cylinder [m].
    num_control_volumes (int): Number of control volumes along the z-axis.

    Returns:
    z_values (numpy array): Array of z positions (center of each control volume) along the height of the cylinder.
    power_density (numpy array): Array of power densities corresponding to each control volume.
    q0 (float): The peak power density constant.
    """
    
    # Cross-sectional area of the cylinder
    A_cross_section = np.pi * r**2

    # Function for the power density profile along the axial direction
    def q(z, q0):
        return q0 * np.sin(np.pi * z / h)

    # Calculate q0 to ensure total power matches Ptot
    q0 = Ptot / (A_cross_section * (2 * h / np.pi))  # derived from the integral

    # Compute the boundaries and midpoints of each control volume
    z_boundaries = np.linspace(0, h, num_control_volumes + 1)
    z_values = (z_boundaries[:-1] + z_boundaries[1:]) / 2  # Midpoints of control volumes

    # Compute power density at each midpoint
    power_density = q(z_values, q0)

    return z_values, power_density

def guess_power_density_cosine(Ptot, h, r, num_control_volumes):
    """
    Compute the power density in a cylinder with a cosine-shaped profile along the axial direction.

    Parameters:
    Ptot (float): Total power released in the cylinder [W].
    h (float): Total height of the cylinder [m].
    r (float): Radius of the cylinder [m].
    num_control_volumes (int): Number of control volumes along the z-axis.

    Returns:
    z_values (numpy array): Array of z positions (center of each control volume) along the height of the cylinder.
    power_density (numpy array): Array of power densities corresponding to each control volume.
    """
    
    # Cross-sectional area of the cylinder
    A_cross_section = np.pi * r**2

    # Function for the power density profile along the axial direction (cosine shape)
    def q(z, q0):
        return q0 * np.cos(np.pi * z / (2 * h))

    # Calculate q0 to ensure total power matches Ptot
    #q0 = Ptot / (A_cross_section * (h / 2))  # derived from the integral of cosine
    q0 = Ptot*np.pi / (A_cross_section * (h * 2))  # derived from the integral of cosine

    # Compute the boundaries and midpoints of each control volume
    z_boundaries = np.linspace(0, h, num_control_volumes + 1)
    z_values = (z_boundaries[:-1] + z_boundaries[1:]) / 2  # Midpoints of control volumes

    # Compute power density at each midpoint
    power_density = q(z_values, q0)

    return z_values, power_density

def compute_power_densities(integrated_powers, r, Iz, height):
    """
    Convert a list of integrated powers at each axial node into power densities.

    Parameters:
    integrated_powers (numpy array): List of integrated powers at each node [W].
    r (float): Radius of the cylinder [m].
    Iz (int) : number of axial-subdivisions.

    Returns:
    power_densities (numpy array): Power densities at each axial node [W/m^3].
    """

    # Cross-sectional area of the cylinder (constant for all nodes)
    A_cross_section = np.pi * r**2
    z_values = np.linspace(0, height, Iz+1)
    # Calculate the heights (Δz) of each control volume
    dz = np.diff(z_values)  # Heights between adjacent z positions

    # Compute the volumes of each control volume
    volumes = A_cross_section * dz  # Volume of each cylinder segment

    # Calculate the power densities for each node
    power_densities = integrated_powers / volumes

    return power_densities

def guess_flat_power_density(Ptot, h, r, num_control_volumes):
    """
    Return a constant power density across the height of the cylinder.
    """

    vol = np.pi * r**2 * h   # Total volume ?
    power_density = (Ptot / vol) * np.ones(num_control_volumes)   # Constant power density
    #z_values = np.linspace(0, h, num_control_volumes + 1)[:-1] + (h / (2 * num_control_volumes))  # Midpoints of control volumes

    return power_density  # Return z values and constant power density

def run_pyTHM(channelType, pitch, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, massFlowRate, Ptot, axial_p_form, fraction_pow_fuel, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting):
    """
    This function runs the pyTHM simulation for a BWR Pincell equivalent channel with specified parameters.
    channelType: str, type of canal (e.g., 'cylindrical', 'square', etc.) : impact hydraulic diameter calculation
    pitch: float, pitch of the channel (m)
    fuelRadius: float, radius of the fuel pellet (m)
    gapRadius: float, radius of the gap ie inner clad radius (m)
    cladRadius: float, radius of the clad, ie outer clad radius (m)
    height: float, height of the channel (m)
    tInlet: float, inlet temperature of the coolant (K)
    pOutlet: float, outlet pressure of the coolant (Pa)
    massFlowRate: float, mass flow rate of the coolant (kg/s)
    Ptot    : float, total fission power in the fuel rod (W)
    axial_p_form: list, axial power distribution form (normalized to have a mean of 1)
    fraction_pow_fuel: float, fraction of power in the fuel (default 1.0)
    kFuel: float, thermal conductivity of the fuel (W/m·K)
    Hgap: float, heat transfer coefficient for the gap (W/m^2·K)
    kClad: float, thermal conductivity of the clad (W/m·K)
    Iz1: int, number of axial subdivisions in the fuel (for convection in fuel channel)
    If: int, number of radial subdivisions in the fuel (for conduction in fuel rod)
    I1: int, number of radial subdivisions in the clad (for conduction in fuel rod)
    zPlotting: list, axial positions for plotting (optional)
    solveConduction: bool, whether to solve conduction equations (default True)
    frfaccorel: float, factor for fuel radial flow correlation (default 1.0)

    """
    THComponent = pyTHM_solver("BWR Pincell equivalent channel", channelType, pitch, fuelRadius, gapRadius, cladRadius, 
                                height, tInlet, pOutlet, massFlowRate, Ptot, axial_p_form, fraction_pow_fuel, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting, 
                                solveConduction = True, dt = 0, t_tot = 0, frfaccorel = 'Churchill', P2Pcorel = 'lockhartMartinelli', voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THComponent.get_TH_parameters()
    QFUEL = THComponent.convection_sol.get_QFUEL()  # Get the fission power distribution in the fuel
    Q_transfered_to_coolant = THComponent.convection_sol.get_Fission_Power()  # Get the power given to the coolant

    return Teff, Twater, rho, voidFrac, P, U, H, QFUEL, Q_transfered_to_coolant

def get_sine_axial_p_form_over_mesh(h, Iz):
    """
    get sine values over a mesh of Iz points along the height h.
    Parameters:
    h (float): Total height of the cylinder [m].
    Iz (int): Number of axial subdivisions.
    """
    # Compute the boundaries and midpoints of each control volume
    z_boundaries = np.linspace(0, h, Iz + 1)
    z_values = (z_boundaries[:-1] + z_boundaries[1:]) / 2  # Midpoints of control volumes
    # Compute sine values at each midpoint
    sine_values = np.sin(np.pi * z_values / h) / np.mean(np.sin(np.pi * z_values / h))  # Normalize to have a mean of 1

    return z_values, sine_values

def get_cosine_axial_p_form_over_mesh(h, Iz):
    """
    get cosine values over a mesh of Iz points along the height h.
    Parameters:
    h (float): Total height of the cylinder [m].
    Iz (int): Number of axial subdivisions.
    """
    # Compute the boundaries and midpoints of each control volume
    z_boundaries = np.linspace(0, h, Iz + 1)
    z_values = (z_boundaries[:-1] + z_boundaries[1:]) / 2  # Midpoints of control volumes
    # Compute cosine values at each midpoint
    cosine_values = np.cos(np.pi * z_values / (2 * h)) / np.mean(np.cos(np.pi * z_values / (2 * h)))  # Normalize to have a mean of 1

    return z_values, cosine_values


if __name__ == "__main__":
    # Define 
    solveConduction = True # Whether to solve conduction equations
    ########## Mesh parameters ###########
    zPlotting = [] #If empty, no plotting of the axial distribution of the fields, otherwise, list of the axial positions where the fields are plotted
    ## Meshing parameters:
    If = 8
    I1 = 3

    # Sensitivity to the meshing parameters
    Iz1 = 20 # number of control volumes in the axial direction, added 70 for comparison with GeN-Foam
    # Iz1 = 10, 20, 40, 50, 70, 80 and 160 are supported for the DONJON solution


    Ptot = 40.0e3 # W, total fission power in the fuel rod, 40kW, 35kW, 30kW, 25kW, 20kW
    # Test powers 40kW, 35kW, 30kW, 25kW, 20kW.
    fraction_pow_fuel = 1.0 ## Default value in THM: DONJON5 is 0.974

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square" # "square", "cylindrical"
    pitch = 1.295e-2 # m : ATRIUM10 pincell pitch
    fuelRadius = 0.4435e-2 # m : fuel rod radius
    gapRadius = 0.4520e-2 # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius = 0.5140e-2 # m : clad external radius
    height = 3.8 # m : height : 3.8 m : active core height in BWRX-300 SMR, 1.55 m : for GeNFoam comparison.


    ## Fluid parameters

    # T_inlet, T_outlet = 270, 287 Celcius
    tInlet = 270 + 273.15 # K, for BWRX-300 SMR core
    # Nominal operating pressure = 7.2 MPa (abs)
    pOutlet =  7.2e6 # Pa 
    # Nominal coolant flow rate = 1530 kg/s
    massFlowRate = 1530  / (200*91)  # kg/s

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000 # W/m^2.K, Heat transfer coefficient for the gap, assumed very high to ensure conduction is dominant in the gap
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d

    flat_axial_p_form = np.ones(Iz1)  # Flat distribution, normalized to have a mean of 1
    #sine_axial_p_form = np.sin(np.linspace(0, np.pi, Iz1))/np.mean(np.sin(np.linspace(0, np.pi, Iz1)))  # Sine distribution, normalized to have a mean of 1
    #cosine_axial_p_form = np.cos(np.linspace(0, np.pi/2, Iz1))/np.mean(np.cos(np.linspace(0, np.pi/2, Iz1))) /Iz1 # Cosine distribution, normalized to have a mean of 1

    z_mesh, sine_axial_p_form = get_sine_axial_p_form_over_mesh(height, Iz1)  # Get sine values over the mesh
    z_mesh, cosine_axial_p_form = get_cosine_axial_p_form_over_mesh(height, Iz1)  # Get cosine values over the mesh
    
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(z_mesh, Ptot*sine_axial_p_form, label='Sine Power Density', marker='o')
    plt.plot(z_mesh, Ptot*cosine_axial_p_form, label='Cosine Power Density', marker='x')
    plt.plot(z_mesh, Ptot*flat_axial_p_form, label='Flat Power Density', linestyle='--')
    plt.xlabel('Axial Position (m)')
    plt.ylabel('Power Density (W/m^3)')
    plt.title('Power Density Distribution in Cylinder')
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_dir}/power_density_distributions_{Ptot/1000}kW_mesh{int(Iz1)}_h{int(height*100)}cm.png')

    # Run pyTHM for each power distribution shape
    # Sine distribution
    THM_sine_results = run_pyTHM(canalType, pitch, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, massFlowRate, Ptot, sine_axial_p_form, fraction_pow_fuel, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting)
    # Cosine distribution
    THM_cosine_results = run_pyTHM(canalType, pitch, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, massFlowRate, Ptot, cosine_axial_p_form, fraction_pow_fuel, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting)
    # Flat distribution
    THM_flat_results = run_pyTHM(canalType, pitch, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, massFlowRate, Ptot, flat_axial_p_form, fraction_pow_fuel, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting)


    # Plot results :
    # Twater, rho, voidFrac, and P
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(z_mesh, THM_sine_results[1], label='Sine Twater', marker='o')
    axs[0, 0].plot(z_mesh, THM_cosine_results[1], label='Cosine Twater', marker='x')
    axs[0, 0].plot(z_mesh, THM_flat_results[1], label='Flat Twater', marker='s')
    axs[0, 0].set_title('Water Temperature')
    axs[0, 0].set_xlabel('Axial Position (m)')
    axs[0, 0].set_ylabel('Temperature (K)')
    axs[0, 0].legend()
    axs[0, 1].plot(z_mesh, THM_sine_results[2], label='Sine Density', marker='o')
    axs[0, 1].plot(z_mesh, THM_cosine_results[2], label='Cosine Density', marker='x')
    axs[0, 1].plot(z_mesh, THM_flat_results[2], label='Flat Density', marker='s')
    axs[0, 1].set_title('Density')
    axs[0, 1].set_xlabel('Axial Position (m)')
    axs[0, 1].set_ylabel('Density (kg/m^3)')
    axs[1, 0].plot(z_mesh, THM_sine_results[3], label='Sine Void Fraction', marker='o')
    axs[1, 0].plot(z_mesh, THM_cosine_results[3], label='Cosine Void Fraction', marker='x')
    axs[1, 0].plot(z_mesh, THM_flat_results[3], label='Flat Void Fraction', marker='s')
    axs[1, 0].set_title('Void Fraction')
    axs[1, 0].set_xlabel('Axial Position (m)')
    axs[1, 0].set_ylabel('Void Fraction')
    axs[1, 1].plot(z_mesh, THM_sine_results[4], label='Sine Pressure', marker='o')
    axs[1, 1].plot(z_mesh, THM_cosine_results[4], label='Cosine Pressure', marker='x')
    axs[1, 1].plot(z_mesh, THM_flat_results[4], label='Flat Pressure', marker='s')
    axs[1, 1].set_title('Pressure')
    axs[1, 1].set_xlabel('Axial Position (m)')
    axs[1, 1].set_ylabel('Pressure (Pa)')
    for ax in axs.flat:
        ax.grid()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/THM_results_comparison_{Ptot/1000}kW_mesh{int(Iz1)}_h{int(height*100)}cm.png')

    results_df_flat = pd.DataFrame({
        'z (m)': z_mesh,
        'TFuel (K)': THM_flat_results[0],
        'Twater (K)': THM_flat_results[1],
        'Density (kg/m^3)': THM_flat_results[2],
        'Void Fraction': THM_flat_results[3],
        'Pressure (Pa)': THM_flat_results[4],
        'Velocity (m/s)': THM_flat_results[5],
        'Enthalpy (J/kg)': THM_flat_results[6],
        'QFUEL (W/m^3)': THM_flat_results[7],
        'Q_transfered_to_coolant (W/m^3)': THM_flat_results[8],
    })

    results_df_sine = pd.DataFrame({
        'z (m)': z_mesh,
        'TFuel (K)': THM_sine_results[0],
        'Twater (K)': THM_sine_results[1],
        'Density (kg/m^3)': THM_sine_results[2],
        'Void Fraction': THM_sine_results[3],
        'Pressure (Pa)': THM_sine_results[4],
        'Velocity (m/s)': THM_sine_results[5],
        'Enthalpy (J/kg)': THM_sine_results[6],
        'QFUEL (W/m^3)': THM_sine_results[7],
        'Q_transfered_to_coolant (W/m^3)': THM_sine_results[8],
    })

    results_df_cosine = pd.DataFrame({
        'z (m)': z_mesh,
        'TFuel (K)': THM_cosine_results[0],
        'Twater (K)': THM_cosine_results[1],
        'Density (kg/m^3)': THM_cosine_results[2],
        'Void Fraction': THM_cosine_results[3],
        'Pressure (Pa)': THM_cosine_results[4],
        'Velocity (m/s)': THM_cosine_results[5],
        'Enthalpy (J/kg)': THM_cosine_results[6],
        'QFUEL (W/m^3)': THM_cosine_results[7],
        'Q_transfered_to_coolant (W/m^3)': THM_cosine_results[8],
    })
    # Save the results to CSV files
    results_df_flat.to_csv(f'{save_dir}/results_df_flat_{int(Ptot)}W_mesh{int(Iz1)}_h{int(height*100)}cm.csv', index=False)
    results_df_sine.to_csv(f'{save_dir}/results_df_sine_{int(Ptot)}W_mesh{int(Iz1)}_h{int(height*100)}cm.csv', index=False)
    results_df_cosine.to_csv(f'{save_dir}/results_df_cosine_{int(Ptot)}W_mesh{int(Iz1)}_h{int(height*100)}cm.csv', index=False)

