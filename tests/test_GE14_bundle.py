import os
import sys
import math
import numpy as np
import pytest
from starterDD.DDModel.DonjonModel import CoreModel
from starterDD.GeometryAnalysis.cartesian_geometry_analysis import CartesianGeometricAnalyser
from pyTHM.Solver.main import pyTHM_solver 
from pyTHM.WaterProperties.waterProperties import FastIAPWS
from conftest import GE14_DATA

def generate_case_name(pdrop, power_kw, profile_type):
    nom = "dfm"
    if pdrop == 1: nom += "p"
    nom += str(int(power_kw))
    if profile_type.lower() in ['cosine', 'cos', 'c']: nom += "c"
    elif profile_type.lower() in ['sinus', 'sin', 's']: nom += "s"
    else: nom += "u"
    nom += "_v"
    return nom

def generate_power_profile(profile_type, nz):
    profil = []
    for i in range(nz):
        z_norm = (i + 0.5) / nz 
        if profile_type == 'cosine':
            val = math.cos((math.pi / 2.0) * z_norm) 
        elif profile_type == 'sinus':
            val = math.sin(math.pi * z_norm)
        else:
            val = 1.0
        profil.append(val)
    
    moyenne = sum(profil) / nz
    return np.array([v / moyenne for v in profil])

@pytest.fixture(scope='session')
def fast_iapws():
    pOutlet = 7.20E+06
    return FastIAPWS(pOutlet)

@pytest.fixture(scope='session')
def solver_data():
    yaml_path = os.path.join(GE14_DATA , "GEOM_BUNDLE.yaml")
        
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Input geometry YAML file {yaml_path} not found.")
    
    # 1. Séparer le dossier et le nom du fichier
    path_to_configs = os.path.dirname(yaml_path)
    core_desc_file = os.path.basename(yaml_path)
    
    # 2. Instancier le modèle du cœur
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
    nz = solver_data["active_flow_data"]["number_of_axial_meshes"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- 2. Matrice de tests ---
    power_kw = 40.0
    power_profile = "cosine"
    pdrop = 1

    
    # Correspondance des options de perte de charge avec les arguments pyTHM
    frfaccorel_choice = 'Churchill'
    
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_Ozaki" 
    axial_pform = generate_power_profile(power_profile, nz)
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    
    # --- 3. Appel direct à pyTHM ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data = solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        axial_p_form=axial_pform,
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
    
    # Récupération et affichage d'un résumé des résultats
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    print(f"Test {case_name}: final void fraction = {voidFrac[-1]:.4f}, DeltaP = {P[0] - P[-1]:.1f} Pa\n")
    deltaP = (P[0] - P[-1])
    assert voidFrac[-1] == pytest.approx(0.76304, abs=1e-5)
    assert deltaP == pytest.approx(41933, abs=1)

def test_GE14_dfmp40c_v_active_no_wr_Ozaki(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    nz = solver_data["active_flow_data"]["number_of_axial_meshes"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- 2. Matrice de tests ---
    power_kw = 40.0
    power_profile = "cosine"
    pdrop = 1

    
    # Correspondance des options de perte de charge avec les arguments pyTHM
    frfaccorel_choice = 'Churchill'
    
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_no_wr_Ozaki" 
    axial_pform = generate_power_profile(power_profile, nz)
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    
    # --- 3. Appel direct à pyTHM ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        axial_p_form=axial_pform,
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
    
    # Récupération et affichage d'un résumé des résultats
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    print(f"Test {case_name}: final void fraction = {voidFrac[-1]:.4f}, DeltaP = {P[0] - P[-1]:.1f} Pa\n")
    deltaP = (P[0] - P[-1])

    assert voidFrac[-1] == pytest.approx(0.74815, abs=1e-5)
    assert deltaP == pytest.approx(42782, abs=1)

def test_GE14_dfmp20c_v_active_wr_noholes_GERamp(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    nz = solver_data["active_flow_data"]["number_of_axial_meshes"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- 2. Matrice de tests ---
    power_kw = 20.0
    power_profile = 'cosine'
    pdrop = 1
    frfaccorel_choice = 'Churchill' 
                
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_noholes_GERamp" 
    axial_pform = generate_power_profile(power_profile, nz)
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    # --- 3. Appel direct à pyTHM ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        axial_p_form=axial_pform,
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
    
    # Récupération et affichage d'un résumé des résultats
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    print(f"Test {case_name}: final void fraction = {voidFrac[-1]:.5f}, DeltaP = {P[0] - P[-1]:.1f} Pa\n")
    deltaP = P[0] - P[-1]
    assert voidFrac[-1] == pytest.approx(0.49882, abs=1e-5)
    assert deltaP == pytest.approx(19682, abs=1)

def test_GE14_dfmp40c_no_holes_WR_Hibiki(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    nz = solver_data["active_flow_data"]["number_of_axial_meshes"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- 2. Matrice de tests ---
    power_kw = 40.0
    power_profile = 'cosine'
    pdrop = 1
    frfaccorel_choice = 'Churchill' 
        
                
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_noholes_HibikiAlSaif" 
    axial_pform = generate_power_profile(power_profile, nz)
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    print(f"Lancement du cas : {case_name}")
    
    # --- 3. Appel direct à pyTHM ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        axial_p_form=axial_pform,
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
    
    # Récupération et affichage d'un résumé des résultats
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    print(f"Test {case_name}: final void fraction = {voidFrac[-1]:.4f}, DeltaP = {P[0] - P[-1]:.1f} Pa\n")
    deltaP = P[0] - P[-1]
    assert voidFrac[-1] == pytest.approx(0.64586, abs=1e-5)
    assert deltaP == pytest.approx(33292, abs=1)

def test_GE14_dfmp20s_WR_EPRIVoidModel(solver_data, fast_iapws):

    ref_acool = solver_data["active_flow_data"]["reference_coolant_cross_sectional_area"]
    nz = solver_data["active_flow_data"]["number_of_axial_meshes"]
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- 2. Matrice de tests ---
    power_kw = 20.0
    power_profile = 'sine'
    pdrop = 1
    frfaccorel_choice = 'Churchill'
        
                
    case_name = generate_case_name(pdrop, power_kw, power_profile)
    case_name = case_name + "_wr_EPRIvoidModel"
    axial_pform = generate_power_profile(power_profile, nz)
    power_w = power_kw * 1000.0 # Conversion kW -> W
    
    
    # --- 3. Appel direct à pyTHM ---
    THsolve = pyTHM_solver(
        case_name=case_name,
        channel_type="square",
        geometric_data=solver_data,
        tInlet=543.15,
        pOutlet=7.20E+06,
        qFlow=mass_flow,
        Powtot=power_w,
        axial_p_form=axial_pform,
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
    
    # Récupération et affichage d'un résumé des résultats
    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    print(f"Test {case_name}: final void fraction = {voidFrac[-1]:.5f}, DeltaP = {P[0] - P[-1]:.1f} Pa\n")
    deltaP = P[0] - P[-1]
    assert voidFrac[-1] == pytest.approx(0.46381, abs=1e-5)
    assert deltaP == pytest.approx(17391, abs=1)