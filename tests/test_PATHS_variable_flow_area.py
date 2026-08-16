import numpy as np
from pyTHM.Solver.main import pyTHM_solver
from pyTHM.WaterProperties.waterProperties import FastIAPWS
import pytest


I_z = 28                     
L_tot = 4.53e-3              
flow_rate = 9.506            
P_out = 7.18e6               
T_in = 550.0                 

A_narrow = 0.010334
A_wide = 0.10334

@pytest.fixture
def fast_iapws():
    return FastIAPWS(P_outlet=P_out)


def test_PATHS_expansion(fast_iapws):
    # ==========================================
    # Base parameters from "The modeling of advanced BWR fuel designs with the NRC fuel depletion codes PARCS/PATHS" 
    # A. Wysocki, A. Ward, A. Manera, T. Downar, Y. Xu, J. March-Leuba, C. Thurston, N. Hudson, A. Ireland
    # in "Nuclear Technology" - Published in Jul. 2015
    # (Table II)
    # ==========================================

    # ==========================================
    # Configuration amont / aval selon le cas
    # ==========================================
    # Le fluide passe de étroit à large
    acools = np.array([A_narrow] * 20 + [A_wide] * 8)
    A_amont = A_narrow
    A_aval = A_wide
    
    # Borda-Carnot corrigé pour être calibré sur la vitesse AVAL
    # Formule classique (sur amont) : (1 - A_amont/A_aval)^2
    # Correction pour aval : on multiplie par (A_aval/A_amont)^2
    K_internal = (A_aval / A_amont - 1.0)**2
    
    rsin_val = A_amont / A_aval

    dhs = np.sqrt(4 * acools / np.pi)
    phs = np.ones(I_z)
    porosities = np.ones(I_z)

    geometric_data = {}

    geometric_data["fuel_data"] = {}
    geometric_data["fuel_data"]["fuel_radius"] = 0.000
    geometric_data["fuel_data"]["gap_radius"] = 0.000
    geometric_data["fuel_data"]["clad_radius"] = 0.000
    geometric_data["fuel_data"]["pin_pitch"] = 0.000
    geometric_data["fuel_data"]["max_rod_length"] = L_tot

    geometric_data["active_flow_data"] = {}
    geometric_data["active_flow_data"]["number_of_axial_meshes"] = I_z
    geometric_data["active_flow_data"]["porosities"] = porosities
    geometric_data["active_flow_data"]["coolant_cross_sectional_areas"] = acools
    geometric_data["active_flow_data"]["hydraulic_diamters"] = dhs
    geometric_data["active_flow_data"]["heated_perimeters"] = phs
    geometric_data["active_flow_data"]["pitch"] = 0.000

    # Dictionnaire des tests K
    k_tests = {
        'K=0': 0.0, 
        'internal=yes': K_internal, 
        'K=1000': 1000.0
    }
    
    results = {}

    # ==========================================
    # Boucle de résolution
    # ==========================================
    for name, K_val in k_tests.items():
        kexp_profile = np.zeros(I_z + 1)
        kcon_profile = np.zeros(I_z + 1)
        rsin_profile = np.ones(I_z + 1)

        # La face de saut est à l'index 20
        kexp_profile[20] = K_val
        rsin_profile[20] = rsin_val

        geometric_data["active_flow_data"]["k_expansion"] = kexp_profile
        geometric_data["active_flow_data"]["k_contraction"] = kcon_profile
        geometric_data["active_flow_data"]["singular_contraction_ratios"] = rsin_profile

        THsolve = pyTHM_solver(
            case_name=f"PARCS_expansion_{name}",
            water_rod=False,         
            channel_type="square",
            geometric_data=geometric_data,
            tInlet=T_in,
            pOutlet=P_out,
            qFlow=flow_rate,
            Powtot=0.0,              
            axial_p_form=np.ones(I_z),  
            fraction_pow_fuel=1.0,
            k_fuel=3.0, H_gap=10000.0, k_clad=15.0,
            I_f=8, I_c=3,
            plot_at_z=[],
            solveConduction=False,   
            fast_iapws_table=fast_iapws,
            dt=0, t_tot=0,
            frfaccorel='null',       
            P2Pcorel='base',
            voidFractionCorrel='EPRIvoidModel',
            numericalMethod="FVM"
        )

        # Extraction de la perte de charge totale
        P_array = THsolve.convection_sol.P[-1]
        Delta_P = P_array[0] - P_array[-1]
        results[name] = Delta_P

    
    assert results["K=0"] == pytest.approx(-543.0, abs=0.1)
    assert results["internal=yes"] == pytest.approx(-96.4, abs=0.1)
    assert results["K=1000"] == pytest.approx(4970.0, abs=0.1)



def test_PATHS_contraction(fast_iapws):
    # ==========================================
    # Base parameters from "The modeling of advanced BWR fuel designs with the NRC fuel depletion codes PARCS/PATHS" 
    # A. Wysocki, A. Ward, A. Manera, T. Downar, Y. Xu, J. March-Leuba, C. Thurston, N. Hudson, A. Ireland
    # in "Nuclear Technology" - Published in Jul. 2015
    # (Table II)
    # ==========================================


    # ==========================================
    # Configuration amont / aval selon le cas
    # ==========================================
    # Le fluide passe de large à étroit
    acools = np.array([A_wide] * 20 + [A_narrow] * 8)
    A_amont = A_wide
    A_aval = A_narrow
    
    # Formule de l'image (Sudden Contraction)
    # d^2 / D^2 correspond exactement au ratio des aires (A_aval / A_amont)
    d_D_ratio = np.sqrt(A_aval / A_amont)
    A_ratio = A_aval / A_amont
    
    if d_D_ratio <= 0.76:
        K_internal = 0.42 * (1.0 - A_ratio)
    else:
        K_internal = (1.0 - A_ratio)**2
        
    # Le K_internal est natif sur la vitesse de la section aval (étroite).
    # Il n'y a pas de correction de vitesse à ajouter pour le solveur.
    rsin_val = A_aval / A_amont

    dhs = np.sqrt(4 * acools / np.pi)
    phs = np.ones(I_z)
    porosities = np.ones(I_z)

    geometric_data = {}

    geometric_data["fuel_data"] = {}
    geometric_data["fuel_data"]["fuel_radius"] = 0.000
    geometric_data["fuel_data"]["gap_radius"] = 0.000
    geometric_data["fuel_data"]["clad_radius"] = 0.000
    geometric_data["fuel_data"]["pin_pitch"] = 0.000
    geometric_data["fuel_data"]["max_rod_length"] = L_tot

    geometric_data["active_flow_data"] = {}
    geometric_data["active_flow_data"]["number_of_axial_meshes"] = I_z
    geometric_data["active_flow_data"]["porosities"] = porosities
    geometric_data["active_flow_data"]["coolant_cross_sectional_areas"] = acools
    geometric_data["active_flow_data"]["hydraulic_diamters"] = dhs
    geometric_data["active_flow_data"]["heated_perimeters"] = phs
    geometric_data["active_flow_data"]["pitch"] = 0.000

    # Dictionnaire des tests K
    k_tests = {
        'K=0': 0.0, 
        'internal=yes': K_internal, 
        'K=1000': 1000.0
    }
    
    results = {}

    # ==========================================
    # Boucle de résolution
    # ==========================================
    for name, K_val in k_tests.items():
        kexp_profile = np.zeros(I_z + 1)
        kcon_profile = np.zeros(I_z + 1)
        rsin_profile = np.ones(I_z + 1)

        # La face de saut est à l'index 20
        kcon_profile[20] = K_val
        rsin_profile[20] = rsin_val

        geometric_data["active_flow_data"]["k_expansion"] = kexp_profile
        geometric_data["active_flow_data"]["k_contraction"] = kcon_profile
        geometric_data["active_flow_data"]["singular_contraction_ratios"] = rsin_profile

        THsolve = pyTHM_solver(
            case_name=f"PARCS_contraction_{name}",
            water_rod=False,         
            channel_type="square",
            geometric_data=geometric_data,
            tInlet=T_in,
            pOutlet=P_out,
            qFlow=flow_rate,
            Powtot=0.0,              
            axial_p_form=np.ones(I_z),  
            fraction_pow_fuel=1.0,
            k_fuel=3.0, H_gap=10000.0, k_clad=15.0,
            I_f=8, I_c=3,
            plot_at_z=[],
            solveConduction=False,   
            fast_iapws_table=fast_iapws,
            dt=0, t_tot=0,
            frfaccorel='null',       
            P2Pcorel='base',
            voidFractionCorrel='EPRIvoidModel',
            numericalMethod="FVM",
        )

        # Extraction de la perte de charge totale
        P_array = THsolve.convection_sol.P[-1]
        Delta_P = P_array[0] - P_array[-1]
        results[name] = Delta_P

    assert results["K=0"]== pytest.approx(584.9, abs=0.1)
    assert results["internal=yes"] == pytest.approx(798.0, abs=0.1)
    assert results["K=1000"] == pytest.approx(564212, abs=1)

