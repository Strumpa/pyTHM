import os
import sys
import numpy as np
import pytest
from starterDD.DDModel.DonjonModel import CoreModel
from starterDD.GeometryAnalysis.cartesian_geometry_analysis import CartesianGeometricAnalyser
from pyTHM.Solver.main import pyTHM_solver 
from pyTHM.WaterProperties.waterProperties import FastIAPWS
from conftest import GE14_DATA

def generate_case_name(pdrop, power_kw, profile_type):
    name = "dfm"
    if pdrop == 1: name += "p"
    name += str(int(power_kw))
    if profile_type.lower() in ['cosine', 'cos', 'c']: name += "c"
    elif profile_type.lower() in ['sine', 'sin', 's']: name += "s"
    else: name += "u"
    name += "_v"
    return name

@pytest.fixture(scope='session')
def fast_iapws():
    pOutlet = 7.20E+06
    return FastIAPWS(pOutlet)

@pytest.fixture(scope='session')
def solver_data():
    yaml_path = os.path.join(GE14_DATA , "GEOM_BUNDLE.yaml")
        
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Input geometry YAML file {yaml_path} not found.")
    
    # 1. Split folder and file name
    path_to_configs = os.path.dirname(yaml_path)
    core_desc_file = os.path.basename(yaml_path)
    
    # 2. Initiate core model
    core_model = CoreModel(
        name="GE14_FUEL_BUNDLE", 
        path_to_yaml_configs=path_to_configs, 
        core_description_yaml=core_desc_file
    )
    
    # 3. Créer les modèles d'assemblage en mémoire
    core_model.createAssemblyModels()
    
    # 4. On passe l'objet `core_model` à l'analyseur
    analyser = CartesianGeometricAnalyser(core_model=core_model, core_i=1, core_j=1)
    # -------------------------

    solver_data = analyser.run_THM_analysis(nz=40, include_water_rods=True) 
    return solver_data

def test_GE14_dfmp40c_v_active_with_wr_Ozaki(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- Test case definition ---
    power_kw = 40.0
    power_profile = "cosine"
    pdrop = 1

    
    # Correspondance des options de perte de charge avec les arguments pyTHM
    frfaccorel_choice = 'Churchill'
    
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_Ozaki" 
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    
    # --- Call pyTHM solver ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data = solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        power_distribution=power_profile,
        fraction_pow_fuel=1.0,
        k_fuel=4.18,
        H_gap=10000.0,
        k_clad=21.5,
        I_f=8,  
        I_c=3,
        water_rod=True,
        water_rod_holes=True,
        plot_at_z=[],
        solveConduction=True,
        fast_iapws_table=fast_iapws,
        dt=0,
        t_tot=0,
        frfaccorel=frfaccorel_choice,
        P2Pcorel='friedel', 
        voidFractionCorrel='Ozaki', 
        numericalMethod="FVM",
    )
    
    # Recover computed parameters and compute pressure drop
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = (P[0] - P[-1])
    assert voidFrac[-1] == pytest.approx(0.76304, abs=1e-5)
    assert deltaP == pytest.approx(41933, abs=1)

def test_GE14_dfmp40c_v_active_no_wr_Ozaki(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- Test case definition ---
    power_kw = 40.0
    power_profile = "cosine"
    pdrop = 1

    
    # Correspondance des options de perte de charge avec les arguments pyTHM
    frfaccorel_choice = 'Churchill'
    
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_no_wr_Ozaki" 
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    
    # --- Call pyTHM solver ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        power_distribution=power_profile,
        fraction_pow_fuel=1.0,
        k_fuel=4.18,
        H_gap=10000.0,
        k_clad=21.5,
        I_f=8,  
        I_c=3,
        plot_at_z=[],
        solveConduction=True,
        water_rod=False,
        water_rod_holes=False,
        fast_iapws_table=fast_iapws,
        dt=0,
        t_tot=0,
        frfaccorel=frfaccorel_choice,
        P2Pcorel='friedel', 
        voidFractionCorrel='Ozaki', 
        numericalMethod="FVM",
    )
    
    # Recover computed parameters and compute pressure drop
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = (P[0] - P[-1])

    assert voidFrac[-1] == pytest.approx(0.74815, abs=1e-5)
    assert deltaP == pytest.approx(42782, abs=1)

def test_GE14_dfmp20c_v_active_wr_noholes_GERamp(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- Test case definition ---
    power_kw = 20.0
    power_profile = 'cosine'
    pdrop = 1
    frfaccorel_choice = 'Churchill' 
                
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_noholes_GERamp" 
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    # --- Call pyTHM solver ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        power_distribution=power_profile,
        fraction_pow_fuel=1.0,
        k_fuel=4.18,
        H_gap=10000.0,
        k_clad=21.5,
        I_f=8,  
        I_c=3,
        water_rod=True,
        water_rod_holes=False,
        plot_at_z=[],
        solveConduction=True,
        fast_iapws_table=fast_iapws,
        dt=0,
        t_tot=0,
        frfaccorel=frfaccorel_choice,
        P2Pcorel='friedel', 
        voidFractionCorrel='GEramp',  
        numericalMethod="FVM",
    )
    
    # Recover computed parameters and compute pressure drop
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert voidFrac[-1] == pytest.approx(0.49882, abs=1e-5)
    assert deltaP == pytest.approx(19682, abs=1)

def test_GE14_dfmp40c_no_holes_WR_Hibiki(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- Test case definition ---
    power_kw = 40.0
    power_profile = 'cosine'
    pdrop = 1
    frfaccorel_choice = 'Churchill' 
        
                
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_noholes_HibikiAlSaif" 
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    # --- Call pyTHM solver ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        power_distribution=power_profile,
        fraction_pow_fuel=1.0,
        k_fuel=4.18,
        H_gap=10000.0,
        k_clad=21.5,
        I_f=8,  
        I_c=3,
        water_rod=True,
        water_rod_holes=False,
        plot_at_z=[],
        solveConduction=True,
        fast_iapws_table=fast_iapws,
        dt=0,
        t_tot=0,
        frfaccorel=frfaccorel_choice,
        P2Pcorel='friedel', 
        voidFractionCorrel='Hibiki_Al-Saif', 
        numericalMethod="FVM"
    )
    
    # Recover computed parameters and compute pressure drop
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert voidFrac[-1] == pytest.approx(0.64586, abs=1e-5)
    assert deltaP == pytest.approx(33292, abs=1)

def test_GE14_dfmp20s_WR_EPRIVoidModel(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- Test case definition ---
    power_kw = 20.0
    power_profile = 'sine'
    pdrop = 1
    frfaccorel_choice = 'Churchill'
        
                
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_EPRIvoidModel"
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    
    # --- Call pyTHM solver ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        power_distribution=power_profile,
        fraction_pow_fuel=1.0,
        k_fuel=4.18,
        H_gap=10000.0,
        k_clad=21.5,
        I_f=8,  
        I_c=3,
        plot_at_z=[],
        solveConduction=True,
        fast_iapws_table=fast_iapws,
        dt=0,
        t_tot=0,
        frfaccorel=frfaccorel_choice,
        P2Pcorel='friedel', 
        voidFractionCorrel='EPRIvoidModel', 
        numericalMethod="FVM",
        water_rod=True,
        water_rod_holes=True
    )
    
    # Recover computed parameters and compute pressure drop
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert voidFrac[-1] == pytest.approx(0.47502, abs=1e-5)
    assert deltaP == pytest.approx(18028, abs=1)