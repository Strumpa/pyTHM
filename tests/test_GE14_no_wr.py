import os
import sys
import math
import numpy as np

# --- Ajustement des chemins pour importer vos modules ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from starterDD.GeometryAnalysis.cartesian_geometry_analysis import CartesianGeometricAnalyser
from pyTHM.Solver.main import pyTHM_solver 

# NOUVEL IMPORT : On importe le CoreModel
from starterDD.DDModel.DonjonModel import CoreModel

def generer_nom_cas(pdrop, power_kw, type_profil):
    nom = "dfm"
    if pdrop == 1: nom += "p"
    nom += str(int(power_kw))
    if type_profil.lower() in ['cosinus', 'cos', 'c']: nom += "c"
    elif type_profil.lower() in ['sinus', 'sin', 's']: nom += "s"
    else: nom += "u"
    nom += "_v"
    return nom

def generer_profil_puissance(type_profil, nz):
    profil = []
    for i in range(nz):
        z_norm = (i + 0.5) / nz 
        if type_profil == 'cosinus':
            val = math.cos((math.pi / 2.0) * z_norm) 
        elif type_profil == 'sinus':
            val = math.sin(math.pi * z_norm)
        else:
            val = 1.0
        profil.append(val)
    
    moyenne = sum(profil) / nz
    return np.array([v / moyenne for v in profil])

def test_GE14_cases():
    starterdd_root = os.path.abspath(os.path.join(root_dir, '..', 'starterDD'))
    yaml_path = os.path.join(starterdd_root, 'data', 'BWRProgressionProblems', 'GE14_inputs', 'CORE', 'GEOM_4x4_mini_CORE.yaml')
    
    if not os.path.exists(yaml_path):
        print(f"Erreur : Fichier {yaml_path} introuvable.")
        return

    print("--- Initialisation du modèle et de l'analyseur géométrique ---")
    
    # --- MODIFICATIONS ICI ---
    # 1. Séparer le dossier et le nom du fichier
    path_to_configs = os.path.dirname(yaml_path)
    core_desc_file = os.path.basename(yaml_path)
    
    # 2. Instancier le modèle du cœur
    core_model = CoreModel(
        name="GE14_mini_core", 
        path_to_yaml_configs=path_to_configs, 
        core_description_yaml=core_desc_file
    )
    
    # 3. Créer les modèles d'assemblage en mémoire
    core_model.createAssemblyModels()
    
    # 4. On passe l'objet `core_model` à l'analyseur
    analyser = CartesianGeometricAnalyser(core_model=core_model, core_i=1, core_j=1)
    # -------------------------

    nz = 40
    
    # --- 1. Extraction de la géométrie ---
    z_min, maxh = analyser.get_z_global_bounds() # en cm
    dz = (maxh - z_min) / nz
    _, pitch_cm = analyser.get_x_global_bounds()

    # On appelle l'analyseur pour tout l'assemblage (cellule 'cv')
    geom_profiles = analyser.execute_profile_z(
        ['cv', [0, 0, pitch_cm, pitch_cm]], 
        dz, dz, z_min, maxh
    )
    
    # Récupération des listes (on ignore kexp, kcon et rsin pour l'instant)
    porosities_profile = geom_profiles[1]
    acool_profile = [a / 10000.0 for a in geom_profiles[2]] # Conversion cm² -> m²
    dhs_profile = [dh * 1e-2 for dh in geom_profiles[3]]    # Conversion cm -> m
    phs_profile = [pch * 1e-2 for pch in geom_profiles[4]]  # Conversion cm -> m
    kexp_profile = geom_profiles[5]
    kcon_profile = geom_profiles[6]
    rsin_profile = geom_profiles[7]

    # --- MODIFICATION ICI : Récupération depuis l'objet ---
    # Paramètres géométriques de base extraits de l'objet DRAGON
    pin_geom = analyser.slices_data[0]['dragon_assembly_model'].pin_geometry_dict
    fuel_radius = pin_geom['fuel_radius'] * 1E-2
    gap_radius = pin_geom['gap_radius'] * 1E-2
    clad_radius = pin_geom['clad_radius'] * 1E-2
    pin_pitch = pin_geom['pin_pitch'] * 1E-2
    pitch_m = pitch_cm * 1E-2
    fuel_rod_length = (maxh - z_min) * 1E-2
    # ------------------------------------------------------

    # Logique de débit massique conservée depuis l'ancien script
    ref_acool = min(acool_profile)
    mass_flow = 8.407E-02 * (ref_acool / 8.470E-05)

    # --- 2. Matrice de tests ---
    puissances_a_tester = [10.0, 20.0, 40.0]
    profils_a_tester = ['cosinus', 'sinus']
    pdrop_options = [0, 1]

    print("--- Lancement des calculs pyTHM en mémoire ---")
    
    for pdrop in pdrop_options:
        # Correspondance des options de perte de charge avec les arguments pyTHM
        frfaccorel_choice = 'Churchill' if pdrop == 1 else 'null'
        
        for power_kw in puissances_a_tester:
            for profil_type in profils_a_tester:
                
                case_name = generer_nom_cas(pdrop, power_kw, profil_type)
                axial_pform = generer_profil_puissance(profil_type, nz)
                power_w = power_kw * 1000.0 # Conversion kW -> W
                
                print(f"Lancement du cas : {case_name}")
                
                # --- 3. Appel direct à pyTHM ---
                THsolve = pyTHM_solver(
                    case_name=case_name,
                    canal_type="square",
                    canal_radius=pitch_m / 2.0, # Demi-côté pour un canal carré
                    fuel_radius=fuel_radius,
                    gap_radius=gap_radius,
                    clad_radius=clad_radius,
                    pin_pitch = pin_pitch,
                    fuel_rod_length=fuel_rod_length,
                    tInlet=543.15,
                    pOutlet=7.20E+06,
                    qFlow=mass_flow,
                    Powtot=power_w,
                    axial_p_form=axial_pform,
                    fraction_pow_fuel=1.0,
                    k_fuel=4.18,
                    H_gap=10000.0,
                    k_clad=21.5,
                    I_z=nz,
                    I_f=8,  
                    I_c=3,
                    plot_at_z=[],
                    solveConduction=True,
                    dt=0,
                    t_tot=0,
                    frfaccorel=frfaccorel_choice,
                    P2Pcorel='friedel', 
                    voidFractionCorrel='Ozaki', 
                    numericalMethod="FVM",
                    porosities=porosities_profile,
                    acools=acool_profile,
                    dhs=dhs_profile,
                    phs=phs_profile,
                    kexp_profile=kexp_profile,
                    kcon_profile=kcon_profile,
                    rsin_profile=rsin_profile,
                    water_rod=False,
                )
                
                # Récupération et affichage d'un résumé des résultats
                Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
                print(f"✔️ Succès {case_name}: Taux de vide final = {voidFrac[-1]:.4f}, DeltaP = {P[0] - P[-1]:.1f} Pa\n")

if __name__ == "__main__":
   test_GE14_cases()