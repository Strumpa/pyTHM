#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet, Raphael Guasch, Benjamin Godard


from ..Conduction.conduction import HeatConductionInFuelPin as FDM_Fuel
from ..Convection.convection import DFMclass
from ..Convection.crossflow import compute_crossflow
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
import os
import re

class pyTHM_solver:
    def __init__(self, case_name, channel_type, geometric_data, tInlet, pOutlet, qFlow, Powtot, axial_p_form, fraction_pow_fuel,
                 k_fuel, H_gap, k_clad, I_f, I_c,
                 water_rod = False, water_rod_holes = False,
                 plot_at_z=[], solveConduction=False, fast_iapws_table=None,
                 dt=0, t_tot=0, frfaccorel = 'base', P2Pcorel = 'base', voidFractionCorrel = 'GEramp', numericalMethod= 'FVM'):
        """
        Main constructor for THM case, first set of parameters correspond to channel properties, second set to fuel/gap/clad properties
        The structure followed is : 
        In FVM_ConvectionInCanal class : use a finite volume method to solve heat convection in the channel, then use the Dittus-Boelter correlation to obtain the convective heat transfer coef 
        between the water and the fuel rod's outer surface. This allows to solve for the temperature at this outer surface. 
        Then in the FDM_HeatConductionInFuelPin class, solve for the heat conduction using MCFD method. Compute temperature at the center of the fuel rod.
        Options to plot results can be activated giving an array of z values at which the results should be plotted.

        Attributes:
        - case_name: Name of the case for identification and output files.
        - channel_type: Type of the channel, either 'cylindrical' or 'square'.
        - geometric_data : dictionnary summarizing the "active_flow_data", "water_rod_data" and "fuel_data"
        - tInlet: Inlet temperature of the coolant (K).
        - pOutlet: Outlet pressure of the coolant (Pa).
        - qFlow: Mass flow rate of the coolant (kg/s).
        - Powtot: Total power generated in a fuel rod (W).
        - axial_p_form: Axial power form factor, representing the power distribution along the axial dimension of the fuel rod.
        - fraction_pow_fuel: Fraction of the total power that is deposited in the fuel.
        - k_fuel: Thermal conductivity of the fuel (W/m/K).
        - H_gap: Heat transfer coefficient through the gap (W/m^2/K).
        - k_clad: Thermal conductivity of the clad (W/m/K).
        - I_f: Number of mesh elements in the fuel.
        - I_c: Number of mesh elements in the clad.
        - water_rod: boolean to activate coupled active flow - water rod models.
        - water_rod_holes: boolean to activate water rod hole models for singular pressure losses.
        - plot_at_z: List of axial positions at which to plot the results.
        - solveConduction: Boolean indicating whether to solve the heat conduction problem in the fuel rod.
        - fast_iapws_table : FastIAPWS project instanced a - priori based on outlet pressure.
        - dt: Time step for transient simulations (s).
        - t_tot: Total simulation time for transient simulations (s).
        - frfaccorel: Friction factor correlation to use in the convection solver ('Churchill' is recommended for a wide range of Reynolds numbers).
        - P2Pcorel: Two phase multiplier ('lockhartMartinelli', 'friedel', 'HEM1', 'HEM2', 'MNmodel').
        - voidFractionCorrel: Drift flux model to use in the convection solver ('GEramp', 'Ozaki', 'EPRIvoidModel', 'HEM', 'modBestion').
        - numericalMethod: Numerical method to use in the convection solver ('FVM' for Finite Volume Method, 'FDM' for Finite Difference Method).
        """
        # General parameters
        self.name = case_name
        self.channel_type = channel_type # cylindrical or square, used to determine the cross sectional flow area in the channel and the hydraulic diameter
        self.water_rod = water_rod

        # time atributes to prepare for transient simulations
        self.t0 = 0
        self.dt = dt
        self.t_end = t_tot

        # Power distribution parameters : 
        self.Powtot = Powtot # Total reactor power in W
        self.axial_pow_form = axial_p_form # axial power form factors, representing the power distribution along the axial dimension of the fuel rod, used to compute the fission power in the fuel rod.
        self.Fpow = fraction_pow_fuel # fraction of the total power that is deposited in the fuel, used to compute the fission power in the fuel rod.

        # thermal solver boundary conditions and thermo-physical data
        # convection problem
        self.tInlet = tInlet
        self.qFlow = qFlow #  mass flux in kg/s, assumed to be constant along the axial profile.
        self.pOutlet =  pOutlet #Pa
        self.FAST_IAPWS = fast_iapws_table
        self.rhoInlet = self.FAST_IAPWS.get_rhol(self.pOutlet)
        # conduction problem
        self.k_fuel = k_fuel # thermal conductivity coefficient in fuel W/m/K
        self.H_gap = H_gap # Heat transfer coefficient through gap W/m^2/K
        self.k_clad = k_clad # thermal conductivity coefficient in clad W/m/K

        # meshing parameters : 
        self.I_z = geometric_data["active_flow_data"]["number_of_axial_meshes"] # number of mesh elements on axial mesh
        self.I_f = I_f # number of mesh elements in the fuel
        self.I_c = I_c # number of mesh elements in clad

        ## Unpack geometric parameters
        # active coolant channel attributes
        porosities = geometric_data["active_flow_data"]["porosities"]
        acools = geometric_data["active_flow_data"]["coolant_cross_sectional_areas"]
        dhs = geometric_data["active_flow_data"]["hydraulic_diamters"]
        phs = geometric_data["active_flow_data"]["heated_perimeters"]
        kexp_profile = geometric_data["active_flow_data"]["k_expansion"]
        kcon_profile = geometric_data["active_flow_data"]["k_contraction"] 
        rsin_profile =  geometric_data["active_flow_data"]["singular_contraction_ratios"] 
        self.pitch = geometric_data["active_flow_data"]["pitch"]

        # water rod attributes
        if water_rod:
            acools_wr = geometric_data["water_rod_data"]["moderator_cross_sectional_areas"]
            porosities_wr = geometric_data["water_rod_data"]["porosities"]
            dhs_wr = geometric_data["water_rod_data"]["hydraulic_diamters"]
            kexp_wr = geometric_data["water_rod_data"]["k_expansion"]
            p_wr = geometric_data["water_rod_data"]["permieters"]
            rwall_wr = geometric_data["water_rod_data"]["thermal_resistances"]
            if water_rod_holes:
                Idelchik_enter = geometric_data["water_rod_data"]["Idelchik_enter"]
                Idelchik_exit = geometric_data["water_rod_data"]["Idelchik_exit"]
                hole_A = geometric_data["water_rod_data"]["hole_A"]
                hole_Z = geometric_data["water_rod_data"]["hole_Z"]
            else: 
                Idelchik_enter = []
                Idelchik_exit = []
                hole_A = []
                hole_Z = []
        else:
            acools_wr = None
            porosities_wr = None
            dhs_wr = None
            kexp_wr = None
            p_wr = None
            rwall_wr = None
            Idelchik_enter = None
            Idelchik_exit = None
            hole_A = None
            hole_Z = None
        
        # fuel geometrical parameters
        self.fuel_radius = geometric_data["fuel_data"]["fuel_radius"] # fuel pin radius in meters
        self.gap_radius = geometric_data["fuel_data"]["gap_radius"] # gap radius in meters, used to determine mesh elements for constant surface discretization
        self.clad_radius = geometric_data["fuel_data"]["clad_radius"] # clad radius in meters, used to determine mesh elements for constant surface discretization
        self.pin_pitch = geometric_data["fuel_data"]["pin_pitch"] # distance between the centers of two adjacent fuel rods in meters
        self.fuel_length = geometric_data["fuel_data"]["max_rod_length"]

        # estimate uInlet in the active flow based on mass flow rate, first estimated for rhoInlet and the coolant cross sectional area.
        self.uInlet = self.qFlow / (self.rhoInlet*acools[0]) #m/s

        # solver options :
        self.frfaccorel = frfaccorel # friction factor correlation
        self.P2Pcorel = P2Pcorel # pressure drop correlation
        self.voidFractionCorrel = voidFractionCorrel # void fraction correlation
        self.numericalMethod = numericalMethod # numerical method used to solve the convection problem in the channel

        self.plot_results = plot_at_z
        self.solveConduction = solveConduction

        print(f"$$$---------- THM: prototype, case treated : {self.name}.")
        if self.dt == 0 :
            self.transient = False
            print("$$$---------- THM: prototype, steady state case.")

        # Prepare and solve 1D heat convection along the z direction in the channel.
        print("$$---------- Calling DFM class.")
        print(f"Setting up heat convection solution along the axial dimension. zmax = {self.fuel_length} m with {self.I_z} axial elements.")
        # Create an object of the class DFMclass
        print(f'self.I_z: {self.I_z}')
        print(f'self.qFlow: {self.qFlow}')
        print(f'self.pOutlet: {self.pOutlet}')
        print(f'self.fuel_length: {self.fuel_length}')
        print(f'self.fuel_radius: {self.fuel_radius}')
        print(f'self.clad_radius: {self.clad_radius}')
        print(f'self.pitch: {self.pitch}')
        print(f'self.Dz: {self.fuel_length/self.I_z}')
        print(f'self.dt: {self.dt}')
        print(f'Courant number: {self.uInlet*self.dt/(self.fuel_length/self.I_z)}')
        print(f"Numerical Method {numericalMethod}")
        self.convection_sol = DFMclass(self.channel_type, self.I_z, self.tInlet, self.qFlow, self.pOutlet, self.fuel_length, self.fuel_radius, self.clad_radius, self.pin_pitch, self.pitch, self.numericalMethod, self.frfaccorel, self.P2Pcorel, self.voidFractionCorrel, self.FAST_IAPWS, dt = self.dt, t_tot = self.t_end, porosities = porosities, acools=acools, dhs = dhs, phs = phs, kexp=kexp_profile, kcon=kcon_profile, rsin=rsin_profile)
        print(f'Hydraulic diameter: {self.convection_sol.D_h}')
        
        Dz_local = self.fuel_length / self.I_z
        z_cells = np.linspace(Dz_local/2, self.fuel_length - Dz_local/2, self.I_z)
        if self.water_rod:
            hole_z_indices = []
            if hole_Z is not None:
                for z in hole_Z:
                    idx = (np.abs(z_cells - z)).argmin()
                    hole_z_indices.append(idx)
            else:
                hole_A = []

            alpha = 0.05 # Initial guess : 5% of global mass flow rate
            alpha_prev = 0.04 
            delta_P_prev = None

            for secant_iter in range(15): # Max 15 attempts to balance inlet pressures
                qFlow_wr = alpha * qFlow
                qFlow_actif = (1 - alpha) * qFlow
                
                print(f"\n--- Sécante {secant_iter} : alpha = {alpha:.4f} (WR: {qFlow_wr:.2f} kg/s, Active: {qFlow_actif:.2f} kg/s) ---")

                v_lat_prev = np.zeros(len(hole_z_indices))
                v_lat_prev_prev = np.zeros(len(hole_z_indices))
                omega_dyn = 0.9
                S_mass_a, S_mom_a, S_h_a = np.zeros(self.I_z+1), np.zeros(self.I_z+1), np.zeros(self.I_z+1)
                S_mass_w, S_mom_w, S_h_w = np.zeros(self.I_z+1), np.zeros(self.I_z+1), np.zeros(self.I_z+1)

                for ping_pong in range(40): 
                    
                    DFM_actif = DFMclass(channel_type, self.I_z, tInlet, qFlow_actif, pOutlet, self.fuel_length, 
                                        self.fuel_radius, self.clad_radius, self.pin_pitch, self.pitch, numericalMethod, 
                                        frfaccorel, P2Pcorel, voidFractionCorrel, self.FAST_IAPWS,
                                        dt=dt, t_tot=t_tot, porosities=porosities, acools=acools, 
                                        dhs=dhs, phs=phs, kexp=kexp_profile, kcon=kcon_profile, rsin=rsin_profile)
                    DFM_actif.set_Fission_Power(Powtot, axial_p_form, fraction_pow_fuel)
                    DFM_actif.update_sources(S_mass_a, S_mom_a, S_h_a)
                    DFM_actif.resolveDFM()

                    kcon_wr_safe = np.zeros(self.I_z+1) if kexp_wr is None else np.zeros_like(kexp_wr)
                    rsin_wr_safe = np.ones(self.I_z+1) if kexp_wr is None else np.ones_like(kexp_wr)
                    
                    DFM_wr = DFMclass(channel_type, self.I_z, tInlet, qFlow_wr, pOutlet, self.fuel_length, 
                                    1e-5, 1e-5, self.pin_pitch, self.pitch, numericalMethod, 
                                    frfaccorel, P2Pcorel, voidFractionCorrel, self.FAST_IAPWS,
                                    dt=dt, t_tot=t_tot, porosities=porosities_wr, acools=acools_wr, 
                                    dhs=dhs_wr, phs=p_wr, kexp=kexp_wr, kcon=kcon_wr_safe, rsin=rsin_wr_safe)
                    DFM_wr.set_Fission_Power(0.0, axial_p_form, fraction_pow_fuel)
                    DFM_wr.update_sources(S_mass_w, S_mom_w, S_h_w)
                    DFM_wr.resolveDFM()
                    
                    S_mass_a_new, S_mom_a_new, S_h_a_new, S_mass_w_new, S_mom_w_new, S_h_w_new, v_lat_new = compute_crossflow(
                        DFM_actif, DFM_wr, hole_z_indices, hole_A, Idelchik_enter, Idelchik_exit, rwall_wr, v_lat_prev, self.FAST_IAPWS
                    )
                    v_lat_new = np.clip(v_lat_new, -80.0, 80.0) 
                    if len(v_lat_new) > 0:
                        error_v_lat = np.max(np.abs(v_lat_new - v_lat_prev))
                    else:
                        error_v_lat = 0.0

                    if error_v_lat < 1e-2:
                        print(f"    Ping-Pong converged in {ping_pong + 1} iterations (max error: {error_v_lat:.4f} m/s)")
                        break
                    
                    print(f"    Ping-Pong iter {ping_pong+1}: error = {error_v_lat:.4f} m/s, omega_dyn = {omega_dyn:.3f}, v_lat_new = {v_lat_new}")

                    if ping_pong >= 2 and len(v_lat_new) > 0:
                        delta_actuel = v_lat_new - v_lat_prev
                        delta_ancien = v_lat_prev - v_lat_prev_prev
                        dot_product = np.sum(delta_actuel * delta_ancien)
                        if dot_product <0:
                            omega_dyn = max(0.05, omega_dyn * 0.5)
                        else:
                            omega_dyn = min(1.0, omega_dyn * 1.1)
                    v_lat_prev_prev = v_lat_prev.copy()
                    if ping_pong == 0:
                        v_lat_prev = omega_dyn*v_lat_new
                        S_mass_a = omega_dyn*S_mass_a_new
                        S_mom_a = omega_dyn*S_mom_a_new
                        S_h_a = omega_dyn*S_h_a_new
                        S_mass_w = omega_dyn*S_mass_w_new
                        S_mom_w = omega_dyn*S_mom_w_new
                        S_h_w = omega_dyn*S_h_w_new
                    else:
                        v_lat_prev = omega_dyn*v_lat_new + (1-omega_dyn)*v_lat_prev
                        S_mass_a = omega_dyn*S_mass_a_new + (1-omega_dyn)*S_mass_a
                        S_mom_a = omega_dyn*S_mom_a_new + (1-omega_dyn)*S_mom_a
                        S_h_a = omega_dyn*S_h_a_new + (1-omega_dyn)*S_h_a
                        S_mass_w = omega_dyn*S_mass_w_new + (1-omega_dyn)*S_mass_w
                        S_mom_w = omega_dyn*S_mom_w_new + (1-omega_dyn)*S_mom_w
                        S_h_w = omega_dyn*S_h_w_new + (1-omega_dyn)*S_h_w

                K_local_orifice = 1.42
                rho_in_a = DFM_actif.rhoL[-1][0]
                U_in_a = DFM_actif.U[-1][0]
                P_in_actif = DFM_actif.P[-1][0] 
                r_seo = 0.026 # Side entry orifice radius
                A_seo = np.pi * r_seo**2
                K_orifice_actif = K_local_orifice * (DFM_actif.areaMatrix[0] / A_seo)**2
                DeltaP_orifice_actif = 0.5 * rho_in_a * U_in_a**2 * K_orifice_actif
                P_plenum_actif = P_in_actif + DeltaP_orifice_actif

                rho_in_w = DFM_wr.rhoL[-1][0]
                U_in_w = DFM_wr.U[-1][0]
                P_in_wr = DFM_wr.P[-1][0] 
                r_gicleur_wr = 0.0040
                A_gicleur = np.pi * r_gicleur_wr**2
                K_orifice_wr = K_local_orifice * (DFM_wr.areaMatrix[0] / (2*A_gicleur))**2
                DeltaP_orifice_wr = 0.5 * rho_in_w * U_in_w**2 * K_orifice_wr
                P_plenum_wr = P_in_wr + DeltaP_orifice_wr
                
                delta_P = P_plenum_actif - P_plenum_wr
                
                print(f"Active flow plenum pressure : {P_plenum_actif:.0f} Pa | Water Rod plenum pressure: {P_plenum_wr:.0f} Pa | Difference: {delta_P:.1f} Pa")           

                if abs(delta_P) < 500.0:
                    print(">>> Reached convergence on inlet mass flow rate (alpha) !")
                    break

                if delta_P_prev is not None:
                    if delta_P != delta_P_prev: # Mathematical safety check
                        derivative = (delta_P - delta_P_prev) / (alpha - alpha_prev)
                        alpha_next = alpha - delta_P / derivative
                        alpha_next = max(0.01, min(0.20, alpha_next))
                    else:
                        alpha_next = alpha * 1.01
                else:
                    alpha_next = 0.06 if delta_P > 0 else 0.04
                
                alpha_prev = alpha
                delta_P_prev = delta_P
                alpha = alpha_next
            
            self.convection_sol = DFM_actif
            self.convection_wr = DFM_wr
            self.alpha_final = alpha
        else:
            # --- SIMPLE RESOLUTION (WITHOUT WATER ROD) ---
            print("\n--- Solving without Water Rod model (active coolant flow only) ---")
            qFlow_actif = self.qFlow # 100% of the flow goes to the active channel
            
            # No lateral exchange, therefore sources = 0
            S_mass_a, S_mom_a, S_h_a = np.zeros(self.I_z+1), np.zeros(self.I_z+1), np.zeros(self.I_z+1)
            
            DFM_actif = DFMclass(channel_type, self.I_z, tInlet, qFlow_actif, pOutlet, self.fuel_length, 
                                 self.fuel_radius, self.clad_radius, self.pin_pitch, self.pitch, numericalMethod, 
                                 frfaccorel, P2Pcorel, voidFractionCorrel, self.FAST_IAPWS,
                                 dt=dt, t_tot=t_tot, porosities=porosities, acools=acools, 
                                 dhs=dhs, phs=phs, kexp=kexp_profile, kcon=kcon_profile, rsin=rsin_profile)
                                 
            DFM_actif.set_Fission_Power(Powtot, axial_p_form, fraction_pow_fuel)
            DFM_actif.update_sources(S_mass_a, S_mom_a, S_h_a)
            DFM_actif.resolveDFM()
            
            self.convection_sol = DFM_actif
            self.convection_wr = None  # No water rod
            self.alpha_final = 0.0

        if self.solveConduction:
            self.Tsurf = self.convection_sol.compute_T_surf()

        if self.solveConduction:
            # Prepare and solve 1D radial heat conduction in the fuel rod, given a Clad surface temperature as a bondary condition 
            self.SetupAndSolve_Conduction_at_all_z() # creates a list of Temperature distributions in the fuel rod given a surface temperature computed by solving the conection problem
            self.get_TFuel_rowlands() # compute and store in the T_eff_fuel attribute the effective fuel temperature given by the Rowlands formula
            self.get_Tfuel_surface() # store in the T_fuel_surface attribute the fuel surface temperature computed

            # extend to Twater : adding a mesh point corresponding to the middle of the channel in the plotting array, add rw to the bounds array and add Twater to the results array
            for index_z in range(len(self.convection_sol.z_mesh)):
                self.T_distributions_axial[index_z].extend_to_canal_visu(rw = self.convection_sol.wall_dist, Tw = self.convection_sol.T_water[index_z])

            self.plot_actif_vs_wr()    
            if self.plot_results:
                for z_val in self.plot_results:
                    self.plot_Temperature_at_z(z_val)
    
    def compute_solver_parameters_from_geometry(self):
        """
        From base geometry parameters :
        
        Compute the list of coolant cross sectional areas,
        Compute the list of porosities from base geometry parameters,
        Compute the list of hydraulic diameters,
        Compute the list of heating perimeters,

        Assume constant geometry along the z axis,
        Assume a single fuel rod with, 
        Assume no channel box or water rods are present.
        """
        if self.channel_type=="square":
            ACool = self.pitch**2 - np.pi*self.clad_radius**2
            Dh =  4 * ACool / (2*np.pi * self.clad_radius)

        elif self.channel_type == 'cylindrical':
            ACool = np.pi * (self.pitch/2) ** 2 - np.pi * self.clad_radius ** 2
            Dh = 4 * ACool / (np.pi * self.pitch + np.pi * self.clad_radius*2)

        # List of axially ordered coolant cross sectional areas
        coolant_cross_sectional_areas = np.array(self.I_z * [ACool])
        # List of axially ordered hydraulic diameters
        dhs = np.array(self.I_z * [Dh])
        # List of axially ordered heated perimeters
        phs = np.array(self.I_z * [2*np.pi*self.clad_radius]) 
        # List of axially ordered porosities
        porosities = coolant_cross_sectional_areas / self.pitch**2


        return coolant_cross_sectional_areas, porosities, dhs, phs

    

    def set_transitoire(self, t_tot, Tini, dt):
        self.t_tot, self.dt = t_tot, dt           
        self.N_temps = round(self.t_tot / self.dt) # number of timesteps, must be an integer
        self.T = np.zeros((self.N_temps+1, self.N_vol)) # 2D temperature array.
        self.T[0] = Tini # Tini is a list
        return


    def SetupAndSolve_Conduction_at_all_z(self, transient = False):
        self.T_distributions_axial = []
        for axial_plane_nb in range(self.convection_sol.nCells):
            z = self.convection_sol.z_mesh[axial_plane_nb]
            T_surf = self.convection_sol.T_surf[axial_plane_nb]
            Qfiss = self.convection_sol.get_Fission_Power()[axial_plane_nb]
            self.T_distributions_axial.append(self.run_Conduction_In_Fuel_at_z(z,Qfiss,T_surf, transient))

        return
    
    def run_Conduction_In_Fuel_at_z(self,z,Qfiss_z,T_surf_z, transient = False):
        print(f"$$---------- Setting up FDM_HeatConductionInFuelPin class for z = {z} m, Qfiss(z) = {Qfiss_z} W/m^3 and T_surf(z) = {T_surf_z} K")
        heat_conduction = FDM_Fuel(self.fuel_radius, self.I_f, self.gap_radius, self.clad_radius, self.I_c, Qfiss_z, self.k_fuel, self.k_clad, self.H_gap, z, T_surf_z)
        if transient:
            print(f"Error: transient case not implemented yet")
        else:
            for i in range(1,heat_conduction.N_node-1):

                if i<heat_conduction.I_f-1: # setting Aij and Di values for nodes inside the fuel 
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Di_half(i-1), 
                                        heat_conduction.get_Di_half(i-1)+heat_conduction.get_Di_half(i), 
                                        -heat_conduction.get_Di_half(i), 
                                        heat_conduction.deltaA_f*heat_conduction.Qfiss)
                elif i==heat_conduction.I_f-1: # setting Aij and Di values for last fuel element
                    heat_conduction.set_ADi_cond(i,
                                    -heat_conduction.get_Di_half(i-1),
                                    (heat_conduction.get_Di_half(i-1)+heat_conduction.get_Ei_fuel()),
                                    -heat_conduction.get_Ei_fuel(),
                                    heat_conduction.deltaA_f*heat_conduction.Qfiss)
                elif i==heat_conduction.I_f: # setting Aij and Di values first fuel / gap interface
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Ei_fuel(), 
                                        heat_conduction.get_Ei_fuel()+heat_conduction.get_Gi(), 
                                        -heat_conduction.get_Gi(), 
                                        0)
                elif i == heat_conduction.I_f+1: # setting Aij and Di values second gap / clad interface
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Gi(), 
                                        heat_conduction.get_Fi_gap()+heat_conduction.get_Gi(), 
                                        -heat_conduction.get_Fi_gap(), 
                                        0)
                elif i == heat_conduction.I_f+2: # Treating the first clad element interface with the gap.
                    heat_conduction.set_ADi_cond(i,
                                                -heat_conduction.get_Ei_gap(),
                                                (heat_conduction.get_Di_half(i)+heat_conduction.get_Ei_gap()),
                                                -heat_conduction.get_Di_half(i),
                                                0)
                elif i>heat_conduction.I_f+2 : # setting Aij and Di for all elements in the clad, apart from the last one
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Di_half(i-1), 
                                        heat_conduction.get_Di_half(i-1)+heat_conduction.get_Di_half(i), 
                                        -heat_conduction.get_Di_half(i), 
                                        0)
            A0,Am1 = np.zeros(heat_conduction.N_node), np.zeros(heat_conduction.N_node) 
            A0[:2] = [heat_conduction.get_Di_half(0), -heat_conduction.get_Di_half(0)]
            Am1[-2:] = [-heat_conduction.get_Di_half(heat_conduction.N_node-2), heat_conduction.get_Di_half(heat_conduction.N_node-2)+heat_conduction.get_Ei_clad()]
            D0 = heat_conduction.deltaA_f*heat_conduction.Qfiss
            Dm1 = heat_conduction.get_Ei_clad()*heat_conduction.T_surf
            heat_conduction.set_CL_cond(A0, Am1, D0, Dm1)
            print(f"$---------- Solving for T(r) using the Finite Difference Method, at z = {z}.")
            heat_conduction.solve_T_in_pin()
        return heat_conduction
    
    def get_TFuel_rowlands(self):
        self.T_eff_fuel = np.zeros(self.convection_sol.nCells)
        for i in range(len(self.T_distributions_axial)):
            self.T_distributions_axial[i].compute_T_eff()
            T_eff_z = self.T_distributions_axial[i].T_eff
            self.T_eff_fuel[i] = T_eff_z
        return
    
    def get_Tfuel_surface(self):
        self.T_fuel_surface = np.zeros(self.convection_sol.nCells)
        for i in range(len(self.T_distributions_axial)):
            if len(self.T_distributions_axial[i].T_distrib) == self.T_distributions_axial[i].N_node:
                T_surf_fuel_z = self.T_distributions_axial[i].T_distrib[self.I_f]
            else:
                T_surf_fuel_z = self.T_distributions_axial[i].T_distrib[self.I_f+1]
            self.T_fuel_surface[i] = T_surf_fuel_z
        return

    def plot_Temperature_at_z(self, z_val):
        print(f"$$---------- Plotting Temperature distribution in rod + channel z = {z_val} m")

        if z_val in self.convection_sol.z_mesh:
            plane_index = int(np.where(self.convection_sol.z_mesh==z_val)[0][0])
            Temperature_distrib_to_plot = self.T_distributions_axial[plane_index].T_distrib
            plotting_mesh = self.T_distributions_axial[plane_index].plot_mesh
            radii_at_bounds = self.T_distributions_axial[plane_index].radii_at_bounds
            physical_regions_bounds = self.T_distributions_axial[plane_index].physical_regions_bounds
            plotting_units = self.T_distributions_axial[plane_index].plotting_units
            Tsurf = self.convection_sol.T_surf[plane_index]
            Twater = self.convection_sol.T_water[plane_index]
            Tcenter = self.T_distributions_axial[plane_index].T_center
        else: # Interpolate between nearest z values to obtain Temperature distribution at a given z.
            second_plane_index = np.where(self.convection_sol.z_mesh>z_val)[0][0]
            first_plane_index = second_plane_index-1
            plane_index = (first_plane_index+second_plane_index)/2
            plotting_mesh = self.T_distributions_axial[first_plane_index].plot_mesh
            radii_at_bounds = self.T_distributions_axial[first_plane_index].radii_at_bounds
            physical_regions_bounds = self.T_distributions_axial[first_plane_index].physical_regions_bounds
            Temperature_distrib_to_plot = self.T_distributions_axial[first_plane_index].T_distrib+(z_val-self.convection_sol.z_mesh[first_plane_index])*(self.T_distributions_axial[second_plane_index].T_distrib-self.T_distributions_axial[first_plane_index].T_distrib)/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])
            plotting_units = self.T_distributions_axial[first_plane_index].plotting_units
            Tsurf = self.convection_sol.T_surf[first_plane_index] + (z_val-self.convection_sol.z_mesh[first_plane_index])*(self.convection_sol.T_surf[second_plane_index]-self.convection_sol.T_surf[first_plane_index])/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])
            Twater = self.convection_sol.T_water[first_plane_index] + (z_val-self.convection_sol.z_mesh[first_plane_index])*(self.convection_sol.T_water[second_plane_index]-self.convection_sol.T_water[first_plane_index])/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])
            Tcenter = self.T_distributions_axial[first_plane_index].T_center + (z_val-self.convection_sol.z_mesh[first_plane_index])*(self.T_distributions_axial[second_plane_index].T_center-self.T_distributions_axial[first_plane_index].T_center)/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])

        
        
        if (isinstance(plane_index, int)):
            plane_index_print = plane_index
        else:
            plane_index_print = str(plane_index).split(".")[0]+str(plane_index).split(".")[1]
        colors = ["lime", "bisque", "chocolate", "royalblue"]
        labels = ["Fuel", "Gap", "Clad", "Water"]
        fig_filled, axs = plt.subplots()
        for i in range(len(physical_regions_bounds)-1):
            axs.fill_between(x=radii_at_bounds, y1=(Tcenter+50)*np.ones(len(radii_at_bounds)), y2=(Twater-50)*np.ones(len(radii_at_bounds)),where=(radii_at_bounds>=physical_regions_bounds[i])&(radii_at_bounds<=physical_regions_bounds[i+1]), color = colors[i], label = labels[i])
        axs.scatter(plotting_mesh, Temperature_distrib_to_plot, marker = "D", color="black",s=10, label="Radial temperature distribution in Fuel rod.")
        axs.legend(loc = "best")
        axs.grid()
        axs.set_xlabel(f"Radial position in {plotting_units}")
        axs.set_ylabel(f"Temperature in K")
        axs.set_title(f"Temperature distribution in fuel rod at z = {z_val}, {self.name}")
        fig_filled.savefig(f"{self.name}_Figure_plane{plane_index_print}_colors")
        plt.show()

    def plotColorMap(self):
        print("$$---------- Plotting colormap of temperature distribution in fuel rod.")
        fig, ax = plt.subplots()
        T = []
        for i in range(len(self.T_distributions_axial)):
            T.append(self.T_distributions_axial[i].T_distrib)
        R = self.T_distributions_axial[0].plot_mesh
        Z = self.convection_sol.z_mesh
        plt.xlabel('Rayon (mm)')
        plt.ylabel('Hauteur (m)')
        plt.title('Temperature (K) en fonction du rayon et de la hauteur')
        plt.pcolormesh(R, Z, T, cmap = 'plasma') 
        plt.colorbar()
        plt.show()


    def get_TH_parameters(self):
        if self.solveConduction:
            return np.array(self.T_eff_fuel), np.array(self.convection_sol.T_water), np.array(self.convection_sol.rho[-1]), np.array(self.convection_sol.voidFraction[-1]), np.array(self.convection_sol.P[-1]), np.array(self.convection_sol.U[-1]), np.array(self.convection_sol.H[-1])
        
        else: 
            return [0], self.convection_sol.T_water, self.convection_sol.voidFraction[-1], np.array(self.convection_sol.rho[-1]), self.convection_sol.P[-1], self.convection_sol.U[-1], self.convection_sol.H[-1]


    def plotThermohydraulicParameters(self, visuParam):
        
        if visuParam[0]:
            fig1, ax1 = plt.subplots()
            if self.solveConduction:
                ax1.plot(self.convection_sol.z_mesh, self.T_eff_fuel, label="Fuel temperature")
            ax1.plot(self.convection_sol.z_mesh, self.convection_sol.T_water, label="Coolant temperature")
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Temperature in K")
            ax1.set_title("Temperature distribution pincell")
            ax1.legend(loc="best")

        if visuParam[1]:
            fig2, ax2 = plt.subplots()
            ax2.plot(self.convection_sol.z_mesh, self.convection_sol.voidFraction[-1], label="Void fraction")
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Void fraction")
            ax2.set_title("Void fraction distribution in coolant channel")
            ax2.legend(loc="best")

        if visuParam[2]:
            fig3, ax3 = plt.subplots()
            ax3.plot(self.convection_sol.z_mesh, self.convection_sol.rho[-1], label="Density")
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Density in kg/m^3")
            ax3.set_title("Density distribution in coolant channel")
            ax3.legend(loc="best")

        if visuParam[3]:
            fig4, ax4 = plt.subplots() 
            ax4.plot(self.convection_sol.z_mesh, self.convection_sol.P[-1], label="Pressure")
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Pressure in Pa")
            ax4.set_title("Pressure distribution in coolant channel")

        if visuParam[4]:
            fig5, ax5 = plt.subplots()
            ax5.plot(self.convection_sol.z_mesh, self.convection_sol.U[-1], label="Enthalpy")
            ax5.set_xlabel("Axial position in m")
            ax5.set_ylabel("Velocity in m/s")
            ax5.set_title("Velocity distribution in coolant channel")

        plt.show()
        return

    def plot_actif_vs_wr(self):
        """
        Generates a complete figure of the thermohydraulic parameters.
        If the water rod is activated, compares Active and WR channels. Otherwise, displays the Active channel only.
        """
        import matplotlib.pyplot as plt
        import os

        # Extraction of the axial coordinates (z) of the active channel
        z = self.convection_sol.z_mesh
        has_wr = hasattr(self, 'convection_wr') and self.convection_wr is not None

        # Creation of a figure with 4 rows and 2 columns
        fig, axs = plt.subplots(4, 2, figsize=(16, 20))
        
        if has_wr:
            fig.suptitle(f"Comparaison Thermohydraulique : Actif vs Water Rod\nCas : {self.name}", fontsize=16, fontweight='bold')
            z_wr = self.convection_wr.z_mesh
        else:
            fig.suptitle(f"Profil Thermohydraulique : channel Actif Seul\nCas : {self.name}", fontsize=16, fontweight='bold')

        # --- 1. Pressure ---
        axs[0, 0].plot(z, self.convection_sol.P[-1], label="Actif", color="darkred", linewidth=2)
        if has_wr: axs[0, 0].plot(z_wr, self.convection_wr.P[-1], label="Water Rod", color="salmon", linestyle="--", linewidth=2)
        axs[0, 0].set_title("Pression", fontweight='bold')
        axs[0, 0].set_ylabel("Pression [Pa]")
        axs[0, 0].grid(True, linestyle=':', alpha=0.7)
        axs[0, 0].legend()

        # --- 2. Densities (Mixture and Liquid) ---
        axs[0, 1].plot(z, self.convection_sol.rho[-1], label="Mélange (Actif)", color="indigo", linewidth=2)
        axs[0, 1].plot(z, self.convection_sol.rhoL[-1], label="Liquide (Actif)", color="blue", alpha=0.5)
        if has_wr:
            axs[0, 1].plot(z_wr, self.convection_wr.rho[-1], label="Mélange (WR)", color="mediumpurple", linestyle="--", linewidth=2)
            axs[0, 1].plot(z_wr, self.convection_wr.rhoL[-1], label="Liquide (WR)", color="cyan", linestyle="--", alpha=0.5)
        axs[0, 1].set_title("Densités", fontweight='bold')
        axs[0, 1].set_ylabel("Densité [kg/m³]")
        axs[0, 1].grid(True, linestyle=':', alpha=0.7)
        axs[0, 1].legend()

        # --- 3. Void fraction ---
        axs[1, 0].plot(z, self.convection_sol.voidFraction[-1], label="Actif", color="darkgreen", linewidth=2)
        if has_wr: axs[1, 0].plot(z_wr, self.convection_wr.voidFraction[-1], label="Water Rod", color="lightgreen", linestyle="--", linewidth=2)
        axs[1, 0].set_title("Taux de vide", fontweight='bold')
        axs[1, 0].set_ylabel("Fraction [-]")
        axs[1, 0].grid(True, linestyle=':', alpha=0.7)
        axs[1, 0].legend()

        # --- 4. Thermodynamic quality ---
        axs[1, 1].plot(z, self.convection_sol.xTh[-1], label="Actif", color="darkorange", linewidth=2)
        if has_wr: axs[1, 1].plot(z_wr, self.convection_wr.xTh[-1], label="Water Rod", color="gold", linestyle="--", linewidth=2)
        axs[1, 1].set_title("Titre thermodynamique", fontweight='bold')
        axs[1, 1].set_ylabel("Titre [-]")
        axs[1, 1].grid(True, linestyle=':', alpha=0.7)
        axs[1, 1].legend()

        # --- 5. Fluid temperature ---
        axs[2, 0].plot(z, self.convection_sol.T_water, label="Actif", color="red", linewidth=2)
        if has_wr: axs[2, 0].plot(z_wr, self.convection_wr.T_water, label="Water Rod", color="pink", linestyle="--", linewidth=2)
        axs[2, 0].set_title("Température du fluide", fontweight='bold')
        axs[2, 0].set_ylabel("Température [K]")
        axs[2, 0].grid(True, linestyle=':', alpha=0.7)
        axs[2, 0].legend()

        # --- 6. Velocities (Mixture, Vapor, Liquid) ---
        axs[2, 1].plot(z, self.convection_sol.U[-1], label="Mélange (Actif)", color="black", linewidth=2)
        axs[2, 1].plot(z, self.convection_sol.Ug, label="Vapeur (Actif)", color="red", linewidth=1.5)
        axs[2, 1].plot(z, self.convection_sol.Ul, label="Liquide (Actif)", color="blue", linewidth=1.5)
        if has_wr:
            axs[2, 1].plot(z_wr, self.convection_wr.U[-1], label="Mélange (WR)", color="gray", linestyle="--", linewidth=2)
            axs[2, 1].plot(z_wr, self.convection_wr.Ug, label="Vapeur (WR)", color="salmon", linestyle="--", linewidth=1.5)
            axs[2, 1].plot(z_wr, self.convection_wr.Ul, label="Liquide (WR)", color="cyan", linestyle="--", linewidth=1.5)
        axs[2, 1].set_title("Vitesses des phases", fontweight='bold')
        axs[2, 1].set_ylabel("Vitesse [m/s]")
        axs[2, 1].grid(True, linestyle=':', alpha=0.7)
        axs[2, 1].legend()

        # --- 7. Power profile ---
        axs[3, 0].plot(z, self.convection_sol.q__, label="Actif", color="darkmagenta", linewidth=2)
        if has_wr: axs[3, 0].plot(z_wr, self.convection_wr.q__, label="Water Rod", color="violet", linestyle="--", linewidth=2)
        axs[3, 0].set_title("Profil de puissance (Densité volumique)", fontweight='bold')
        axs[3, 0].set_ylabel("Puissance [W/m³]")
        axs[3, 0].grid(True, linestyle=':', alpha=0.7)
        axs[3, 0].legend()
        
        # --- 8. Empty subplot (to maintain symmetry) ---
        axs[3, 1].axis('off')

        # Add X labels for each subplot
        for ax in axs.flat:
            if ax.has_data():
                ax.set_xlabel("Position axiale z [m]")

        # Adjustment of spacing between subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # --- SAVE LOGIC ---
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        suffix = "actif_vs_wr" if has_wr else "actif_seul"
        filepath = os.path.join(results_dir, f"{self.name}_{suffix}.png")
        
        plt.savefig(filepath, dpi=300)
        print(f">>> Graphique sauvegardé avec succès sous : {filepath}")
        
        plt.close(fig)
    

class plotting:
    def __init__(self, caseList):
        self.caseList = caseList

    def writeResults(self, filename):
    # Create a new Excel workbook
        temperature_cases, void_fraction_cases, density_cases, pressure, velocity, enthalpy_cases = [self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh]
        for case in self.caseList:
            temperature_cases.append(list(case.convection_sol.T_water))
            void_fraction_cases.append(list(case.convection_sol.voidFraction[-1]))
            density_cases.append(list(case.convection_sol.rho[-1]))
            pressure.append(list(case.convection_sol.P[-1]))
            velocity.append(list(case.convection_sol.U[-1]))
            enthalpy_cases.append(list(case.convection_sol.H[-1]))
            parameters = [['waterRadius', 'fuelRadius','gapRadius','cladRadius','height','pOutlet', 'tInlet', 'u_inlet', 'qFlow', 'Iz1', 'qFiss', 'Dz', 'frfaccorel', 'P2Pcorel', 'voidFractionCorrel', 'numericalMethod']]
            for case in self.caseList:
                parameters.append([case.r_w, case.r_f, case.gap_r, case.clad_r, case.Lf, case.pOutlet, case.tInlet, case.uInlet, case.qFlow, case.I_z, case.Qfiss, case.convection_sol.DV, case.frfaccorel, case.P2Pcorel, case.voidFractionCorrel, case.numericalMethod])

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            #Write parameters for each case to a sheet
            parameters_df = pd.DataFrame(parameters).T
            parameters_df.columns = ['Parameter'] + [f"Case {i}" for i in range(len(self.caseList))]
            parameters_df.to_excel(writer, sheet_name='Parameters', index=False)

            #Write temperature data to a sheet
            temperature_df = pd.DataFrame(temperature_cases).T
            temperature_df.columns = ['Axial position'] + [f"Temperature (K) Case {i}" for i in range(1,len(temperature_cases))]
            temperature_df.to_excel(writer, sheet_name='Temperature', index=False)

            #Write void fraction data to a sheet
            void_fraction_df = pd.DataFrame(void_fraction_cases).T
            void_fraction_df.columns =  ['Axial position']+[f"Void Fraction Case {i}" for i in range(1,len(void_fraction_cases))]
            void_fraction_df.to_excel(writer, sheet_name='Void Fraction', index=False)

            #Write density data to a sheet
            density_df = pd.DataFrame(density_cases).T
            density_df.columns =  ['Axial position']+[f"Density (kg/m3) Case {i}" for i in range(1,len(density_cases))]
            density_df.to_excel(writer, sheet_name='Density', index=False)

            #Write pressure data to a sheet
            pressure_df = pd.DataFrame(pressure).T
            pressure_df.columns = ['Axial position']+[f"Pressure (Pa) Case {i}" for i in range(1,len(pressure))]
            pressure_df.to_excel(writer, sheet_name='Pressure', index=False)

            #Write velocity data to a sheet
            velocity_df = pd.DataFrame(velocity).T
            velocity_df.columns =  ['Axial position']+[f"Velocity (m/s) Case {i}" for i in range(1,len(velocity))]
            velocity_df.to_excel(writer, sheet_name='Velocity', index=False)

            #Write enthalpy data to a sheet
            enthalpy_df = pd.DataFrame(enthalpy_cases).T
            enthalpy_df.columns = ['Axial position']+[f"Enthalpy (kg/m3) Case {i+1}" for i in range(1,len(enthalpy_cases))]
            enthalpy_df.to_excel(writer, sheet_name='Enthalpy', index=False)
            
    def compute_error(self, GenFoamPathCase, compParam, genFoamVolumeFraction):

        # Read the Excel file
        df = pd.read_excel(GenFoamPathCase)

        # Create empty lists for each column
        columns = df.columns.tolist()
        data = [[] for _ in columns]

        # Iterate over each row and append values to the corresponding list
        for index, row in df.iterrows():
            for i, col in enumerate(columns):
                data[i].append(row[col])

        for i in range(len(data[7])):
            data[7][i] = (1/(1-genFoamVolumeFraction)) * data[7][i]

        genfoamCASE = [data[0], data[3], data[7], data[3], data[1], data[5]]

        Tw_error, voidFraction_error, pressure_error, velocity_error, = [], [], [], []
        
        for i in range(len(self.caseList)):
            Tw_error.append([])
            voidFraction_error.append([])
            pressure_error.append([])
            velocity_error.append([])
            for j in range(len(self.caseList[i].convection_sol.z_mesh)):
                jGF = int(j*len(genfoamCASE[0])/len(self.caseList[i].convection_sol.z_mesh))
                Tw_error[i].append(100*abs(self.caseList[i].convection_sol.T_water[j] - genfoamCASE[1][jGF])/genfoamCASE[1][jGF])
                voidFraction_error[i].append(100*abs(self.caseList[i].convection_sol.voidFraction[-1][j] - genfoamCASE[2][jGF]))
                pressure_error[i].append(100*abs(self.caseList[i].convection_sol.P[-1][j] - genfoamCASE[4][jGF])/genfoamCASE[4][jGF])
                velocity_error[i].append(100*abs(self.caseList[i].convection_sol.U[-1][j] - genfoamCASE[5][jGF])/genfoamCASE[5][jGF])

        Tw_error_RMS, voidFraction_error_RMS, pressure_error_RMS, velocity_error_RMS, = [], [], [], []

        for i in range(len(self.caseList)):
            Tw_error_RMS.append([])
            voidFraction_error_RMS.append([])
            pressure_error_RMS.append([])
            velocity_error_RMS.append([])
            for j in range(len(self.caseList[i].convection_sol.z_mesh)):
                jGF = int(j*len(genfoamCASE[0])/len(self.caseList[i].convection_sol.z_mesh))
                Tw_error_RMS[i].append(self.caseList[i].convection_sol.T_water[j] - genfoamCASE[1][jGF])
                voidFraction_error_RMS[i].append(self.caseList[i].convection_sol.voidFraction[-1][j] - genfoamCASE[2][jGF])
                pressure_error_RMS[i].append(self.caseList[i].convection_sol.P[-1][j] - genfoamCASE[4][jGF])
                velocity_error_RMS[i].append(self.caseList[i].convection_sol.U[-1][j] - genfoamCASE[5][jGF])

        if compParam == 'numericalMethod':
            models = [self.caseList[i].numericalMethod for i in range(len(self.caseList))]
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].numericalMethod)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].numericalMethod)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].numericalMethod)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].numericalMethod)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        elif compParam == 'voidFractionCorrel':
            models = [self.caseList[i].voidFractionCorrel for i in range(len(self.caseList))]
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].voidFractionCorrel)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].voidFractionCorrel)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].voidFractionCorrel)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].voidFractionCorrel)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        elif compParam == 'frfaccorel':
            models = [self.caseList[i].frfaccorel for i in range(len(self.caseList))]
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].frfaccorel)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].frfaccorel)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].frfaccorel)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].frfaccorel)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()
        
        elif compParam == 'nCells':
            models = [self.caseList[i].convection_sol.nCells for i in range(len(self.caseList))]
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].convection_sol.nCells)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].convection_sol.nCells)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].convection_sol.nCells)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].convection_sol.nCells)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        elif compParam == 'P2Pcorrel':
            models = [self.caseList[i].P2Pcorel for i in range(len(self.caseList))]
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].P2Pcorel)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].P2Pcorel)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].P2Pcorel)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].P2Pcorel)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        
        temperature_errors = {
                'mean': [],
                'min': [],
                'max': []
            }
        pressure_errors = {
                'mean': [],
                'min': [],
                'max': []
            }
        velocity_errors = {
                'mean': [],
                'min': [],
                'max': []
            }
        voidFraction_errors = {
                'mean': [],
                'min': [],
                'max': []
            }
        

        for i in range(len(self.caseList)):
            voidFraction_error[i] = self.cleanList(voidFraction_error[i])
            Tw_error[i] = self.cleanList(Tw_error[i])
            pressure_error[i] = self.cleanList(pressure_error[i])
            velocity_error[i] = self.cleanList(velocity_error[i])
            voidFraction_error_RMS[i] = self.cleanList(voidFraction_error_RMS[i])
            Tw_error_RMS[i] = self.cleanList(Tw_error_RMS[i])
            pressure_error_RMS[i] = self.cleanList(pressure_error_RMS[i])
            velocity_error_RMS[i] = self.cleanList(velocity_error_RMS[i])

            meanVoid = 0
            meanTemp = 0
            meanPressure = 0
            meanVelocity = 0

            for j in range(len(voidFraction_error[i])):
                meanVoid += (voidFraction_error[i][j]**2)
            for j in range(len(Tw_error[i])):
                meanTemp += (Tw_error[i][j]**2)
            for j in range(len(pressure_error[i])):
                meanPressure += (pressure_error[i][j]**2)
            for j in range(len(velocity_error[i])):
                meanVelocity += (velocity_error[i][j]**2)

            meanVoid = np.sqrt(meanVoid/len(voidFraction_error))
            meanTemp = np.sqrt(meanTemp/len(Tw_error))
            meanPressure = np.sqrt(meanPressure/len(pressure_error))
            meanVelocity = np.sqrt(meanVelocity/len(velocity_error))

            voidFraction_errors['mean'].append(np.mean(voidFraction_error[i]))
            voidFraction_errors['max'].append(np.max(voidFraction_error[i]))
            voidFraction_errors['min'].append(np.min(voidFraction_error[i]))
            temperature_errors['mean'].append(np.mean(Tw_error[i]))
            temperature_errors['max'].append(np.max(Tw_error[i]))
            temperature_errors['min'].append(np.min(Tw_error[i]))
            pressure_errors['mean'].append(np.mean(pressure_error[i]))
            pressure_errors['max'].append(np.max(pressure_error[i]))
            pressure_errors['min'].append(np.min(pressure_error[i]))
            velocity_errors['mean'].append(np.mean(velocity_error[i]))
            velocity_errors['max'].append(np.max(velocity_error[i]))
            velocity_errors['min'].append(np.min(velocity_error[i]))

        # Creation of plots for pressure, temperature and velocity
        self.plot_error_graph(models, voidFraction_errors, temperature_errors, pressure_errors, velocity_errors)
        plt.show()

    def cleanList(self, data):
        # Convert to numpy array if it's not already
        data = np.array(data)
        # Filter out NaN and infinite values
        cleaned_data = data[np.isfinite(data)]
        return cleaned_data

    def plotSimple(self):
        fig1, ax1 = plt.subplots()
        for i in range(len(self.caseList)):
            ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].voidFractionCorrel)
        ax1.set_xlabel("Axial position in m")
        ax1.set_ylabel("Temperature in K")
        ax1.set_title("Temperature distribution in pincell")
        ax1.legend(loc="best")


        fig2, ax2 = plt.subplots()
        for i in range(len(self.caseList)):
            ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].voidFractionCorrel)
        ax2.set_xlabel("Axial position in m")
        ax2.set_ylabel("Void fraction")
        ax2.set_title("Void fraction distribution in coolant channel")
        ax2.legend(loc="best")


        fig3, ax3 = plt.subplots()
        for i in range(len(self.caseList)):
            ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].voidFractionCorrel)
        ax3.set_xlabel("Axial position in m")
        ax3.set_ylabel("Density in kg/m^3")
        ax3.set_title("Density distribution in coolant channel")
        ax3.legend(loc="best")


        fig4, ax4 = plt.subplots()
        for i in range(len(self.caseList)):
            ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].voidFractionCorrel)
        ax4.set_xlabel("Axial position in m")
        ax4.set_ylabel("Pressure in Pa")
        ax4.set_title("Pressure distribution in coolant channel")
        ax4.legend(loc="best")
        plt.show()

        fig5, ax5 = plt.subplots()
        for i in range(len(self.caseList)):
            ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].voidFractionCorrel)
        ax5.set_xlabel("Axial position in m")
        ax5.set_ylabel("Velocity in m/s")
        ax5.set_title("Velocity distribution in coolant channel")
        ax5.legend(loc="best")
        plt.show()


    # Function to create plots for the different variables
    def plot_error_graph(self, models, void_fraction_errors, temperature_errors, pressure_errors, velocity_errors):
        fig, ax = plt.subplots()

        # Positions pour chaque groupe de barres
        width = 0.2  # Bar width
        x = np.arange(len(models))  # Model positions

        # Bars for void fraction errors
        bars_void_fraction = ax.bar(x - 1.5*width, void_fraction_errors['mean'], width, 
                                    yerr=[void_fraction_errors['mean'], [void_fraction_errors['max'][i] - void_fraction_errors['mean'][i] for i in range(len(models))]],
                                    capsize=5, label='Fraction de vide', color='skyblue')

        # Bars for temperature errors
        bars_temperature = ax.bar(x - 0.5*width, temperature_errors['mean'], width, 
                                yerr=[temperature_errors['mean'], [temperature_errors['max'][i] - temperature_errors['mean'][i] for i in range(len(models))]],
                                capsize=5, label='Température', color='lightcoral')

        # Bars for pressure errors
        bars_pressure = ax.bar(x + 0.5*width, pressure_errors['mean'], width, 
                            yerr=[pressure_errors['mean'], [pressure_errors['max'][i] - pressure_errors['mean'][i] for i in range(len(models))]],
                            capsize=5, label='Pression', color='lightgreen')

        """ # Barres pour les erreurs de vitesse
        bars_velocity = ax.bar(x + 1.5*width, velocity_errors['mean'], width, 
                            yerr=[velocity_errors['mean'], [velocity_errors['max'][i] - velocity_errors['mean'][i] for i in range(len(models))]],
                            capsize=5, label='Vitesse', color='orange') """

        # Add labels and title
        ax.set_xlabel('Modèles')
        ax.set_ylabel('Erreurs mean/max (%)')
        ax.set_title('Comparaison des erreurs de fraction de vide, température, pression et vitesse')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(loc="upper left")

        # Affichage du graphique
        plt.tight_layout()

    def plotComparison(self, compParam, visuParam):
        if compParam == 'voidFractionCorrel':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].voidFractionCorrel)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].voidFractionCorrel)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant channel")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].voidFractionCorrel)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant channel")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].voidFractionCorrel)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant channel")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].voidFractionCorrel)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant channel")
                ax5.legend(loc="best")

            fig6, ax6 = plt.subplots()
            for i in range(len(self.caseList)):
                ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
            ax6.set_xlabel("Axial position in m")
            ax6.set_ylabel("Fission power in W/m^3")
            ax6.set_title("Fission power distribution in fuel rod")

            
            fig7, ax7 = plt.subplots()
            for i in range(len(self.caseList)):
                ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].voidFractionCorrel)
            ax7.set_xlabel("Axial position in m")
            ax7.set_ylabel("Enthalpy in K")
            ax7.set_title("Enthalpy distribution in pincell")
            ax7.legend(loc="best")

            plt.show()
    
        elif compParam == 'frfaccorel':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].frfaccorel)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].frfaccorel)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant channel")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].frfaccorel)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant channel")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].frfaccorel)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant channel")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].frfaccorel)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant channel")
                ax5.legend(loc="best")

            fig6, ax6 = plt.subplots()
            for i in range(len(self.caseList)):
                ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
            ax6.set_xlabel("Axial position in m")
            ax6.set_ylabel("Fission power in W/m^3")
            ax6.set_title("Fission power distribution in fuel rod")

            
            fig7, ax7 = plt.subplots()
            for i in range(len(self.caseList)):
                ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].frfaccorel)
            ax7.set_xlabel("Axial position in m")
            ax7.set_ylabel("Enthalpy in kJ/kg")
            ax7.set_title("Enthalpy distribution in pincell")
            ax7.legend(loc="best")

            plt.show()
    
        elif compParam == 'P2Pcorrel':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].P2Pcorel)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].P2Pcorel)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant channel")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].P2Pcorel)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant channel")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].P2Pcorel)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant channel")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].P2Pcorel)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant channel")
                ax5.legend(loc="best")

            fig6, ax6 = plt.subplots()
            for i in range(len(self.caseList)):
                ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
            ax6.set_xlabel("Axial position in m")
            ax6.set_ylabel("Fission power in W/m^3")
            ax6.set_title("Fission power distribution in fuel rod")

            
            fig7, ax7 = plt.subplots()
            for i in range(len(self.caseList)):
                ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].P2Pcorel)
            ax7.set_xlabel("Axial position in m")
            ax7.set_ylabel("Enthalpy in K")
            ax7.set_title("Enthalpy distribution in pincell")
            ax7.legend(loc="best")

            plt.show()

        elif compParam == 'numericalMethod':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].numericalMethod)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].numericalMethod)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant channel")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].numericalMethod)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant channel")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].numericalMethod)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant channel")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].numericalMethod)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant channel")
                ax5.legend(loc="best")

            if visuParam[5]:
                fig6, ax6 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
                ax6.set_xlabel("Axial position in m")
                ax6.set_ylabel("Fission power in W/m^3")
                ax6.set_title("Fission power distribution in fuel rod")

            if visuParam[6]:
                fig7, ax7 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].numericalMethod)
                ax7.set_xlabel("Axial position in m")
                ax7.set_ylabel("Enthalpy in K")
                ax7.set_title("Enthalpy distribution in pincell")
                ax7.legend(loc="best")

            plt.show()

    
    def GenFoamComp(self, GenFoamPathCase, compParam, visuParam, genFoamVolumeFraction):

        # Read the Excel file
        df = pd.read_excel(GenFoamPathCase)

        # Create empty lists for each column
        columns = df.columns.tolist()
        data = [[] for _ in columns]

        # Iterate over each row and append values to the corresponding list
        for index, row in df.iterrows():
            for i, col in enumerate(columns):
                data[i].append(row[col])

        for i in range(len(data[7])):
            data[7][i] = (1/(1-genFoamVolumeFraction)) * data[7][i]

        genfoamCASE = [data[0], data[3], data[7], data[4], data[1], data[5]]

        if compParam == 'voidFractionCorrel':
            title = f"Methode numérique: {self.caseList[0].numericalMethod}, \n Correlation multiplicateur biphasique: {self.caseList[0].convection_sol.P2Pcorel}, \n Correlation facteur de friction: {self.caseList[0].convection_sol.frfaccorel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].voidFractionCorrel)
                ax1.step(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.grid()
                ax1.set_title(title)
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].voidFractionCorrel)
                ax2.step(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.grid()
                ax2.set_title(title)
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].voidFractionCorrel)
                ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(title)
                ax3.grid()
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].voidFractionCorrel)
                ax4.step(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(title)
                ax4.grid()
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].voidFractionCorrel)
                ax5.step(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(title)
                ax5.grid()
                ax5.legend(loc="best")

            plt.show()

        if compParam == 'frfaccorel':
            title = f"Methode numérique: {self.caseList[0].numericalMethod}, \n Correlation multiplicateur biphasique: {self.caseList[0].convection_sol.P2Pcorel}, \n Correlation void fraction: {self.caseList[0].convection_sol.voidFractionCorrel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].frfaccorel)
                ax1.plot(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(f"{title}")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].frfaccorel)
                ax2.plot(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(f"{title}")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].frfaccorel)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(f"{title}")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].frfaccorel)
                ax4.plot(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(f"{title}")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].frfaccorel)
                ax5.plot(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(f"{title}")
                ax5.legend(loc="best")

            fig7, ax7 = plt.subplots()
            for i in range(len(self.caseList)):
                ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].frfaccorel)
            ax7.set_xlabel("Axial position in m")
            ax7.set_ylabel("Enthalpy in kJ/kg")
            ax7.set_title("Enthalpy distribution in pincell")
            ax7.legend(loc="best")
            
            

            plt.show()

        if compParam == 'numericalMethod':
            title = f"Correlation fracteur de friction: {self.caseList[0].convection_sol.frfaccorel}, \n Correlation multiplicateur biphasique: {self.caseList[0].convection_sol.P2Pcorel}, \n Correlation void fraction: {self.caseList[0].convection_sol.voidFractionCorrel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].numericalMethod)
                ax1.plot(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(f"{title}")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].numericalMethod)
                ax2.plot(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(f"{title}")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].numericalMethod)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(f"{title}")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].numericalMethod)
                ax4.plot(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(f"{title}")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].numericalMethod)
                ax5.plot(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(f"{title}")
                ax5.legend(loc="best")

            plt.show()

        if compParam == 'P2Pcorrel':
            title = f"Correlation fracteur de friction: {self.caseList[0].convection_sol.frfaccorel}, \n Méthode numérique: {self.caseList[0].convection_sol.numericalMethod}, \n Correlation void fraction: {self.caseList[0].convection_sol.voidFractionCorrel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].P2Pcorel)
                ax1.step(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(f"{title}")
                ax1.grid()
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].P2Pcorel)
                ax2.step(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(f"{title}")
                ax2.grid()
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].P2Pcorel)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(f"{title}")
                ax3.grid()
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].P2Pcorel)
                ax4.step(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.grid()
                ax4.set_title(f"{title}")

                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.step(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].P2Pcorel)
                ax5.step(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(f"{title}")
                ax5.grid()
                ax5.legend(loc="best")

            plt.show()

        if compParam == 'nCells':
            title = f"Correlation fracteur de friction: {self.caseList[0].convection_sol.frfaccorel}, \n Méthode numérique: {self.caseList[0].convection_sol.numericalMethod}, \n Correlation void fraction: {self.caseList[0].convection_sol.voidFractionCorrel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].convection_sol.nCells)
                ax1.plot(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(f"{title}")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].convection_sol.nCells)
                ax2.plot(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(f"{title}")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].convection_sol.nCells)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(f"{title}")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].convection_sol.nCells)
                ax4.plot(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(f"{title}")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].convection_sol.nCells)
                ax5.plot(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(f"{title}")
                ax5.legend(loc="best")

            plt.show()

        