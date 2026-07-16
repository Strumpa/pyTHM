import numpy as np
from pyTHM.Solver.main import pyTHM_solver

def run_parcs_test(case_type):
    # ==========================================
    # Paramètres de base (Table II)
    # ==========================================
    I_z = 28                     
    L_tot = 4.53e-3              
    flow_rate = 9.506            
    P_out = 7.18e6               
    T_in = 550.0                 

    A_narrow = 0.010334
    A_wide = 0.10334

    # ==========================================
    # Configuration amont / aval selon le cas
    # ==========================================
    if case_type == "Expansion":
        # Le fluide passe de étroit à large
        acools = np.array([A_narrow] * 20 + [A_wide] * 8)
        A_amont = A_narrow
        A_aval = A_wide
        
        # Borda-Carnot corrigé pour être calibré sur la vitesse AVAL
        # Formule classique (sur amont) : (1 - A_amont/A_aval)^2
        # Correction pour aval : on multiplie par (A_aval/A_amont)^2
        K_internal = (A_aval / A_amont - 1.0)**2
        
        rsin_val = A_amont / A_aval

    elif case_type == "Contraction":
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

    else:
        raise ValueError("Type de cas inconnu.")

    dhs = np.sqrt(4 * acools / np.pi)
    phs = np.ones(I_z)
    porosities = np.ones(I_z)

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
        if case_type == "Expansion":
            kexp_profile[20] = K_val
        else:
            kcon_profile[20] = K_val
            
        rsin_profile[20] = rsin_val

        THsolve = pyTHM_solver(
            case_name=f"PARCS_{case_type}_{name}",
            water_rod=False,         
            canal_type="square",
            canal_radius=0.065,      
            fuel_radius=0.004,
            gap_radius=0.0041,
            clad_radius=0.0047,
            fuel_rod_length=L_tot,
            tInlet=T_in,
            pOutlet=P_out,
            qFlow=flow_rate,
            Powtot=0.0,              
            axial_p_form=np.ones(I_z),  
            fraction_pow_fuel=1.0,
            k_fuel=3.0, H_gap=10000.0, k_clad=15.0,
            I_z=I_z, I_f=8, I_c=3,
            plot_at_z=[],
            solveConduction=False,   
            dt=0, t_tot=0,
            frfaccorel='null',       
            P2Pcorel='base',
            voidFractionCorrel='EPRIvoidModel',
            numericalMethod="FVM",
            porosities=porosities,
            acools=acools,
            dhs=dhs,
            phs=phs,
            kexp_profile=kexp_profile,
            kcon_profile=kcon_profile,
            rsin_profile=rsin_profile
        )

        # Extraction de la perte de charge totale
        P_array = THsolve.convection_sol.P[-1]
        Delta_P = P_array[0] - P_array[-1]
        results[name] = Delta_P

    return results

# ==========================================
# Exécution du script principal
# ==========================================
if __name__ == "__main__":
    import sys, os
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    res_exp = run_parcs_test("Expansion")
    res_con = run_parcs_test("Contraction")
    
    sys.stdout.close()
    sys.stdout = original_stdout

    print("\n" + "="*50)
    print("  RÉSUMÉ DES PERTES DE CHARGE (PARCS / PATHS)")
    print("="*50)
    
    print("\n1. EXPANSION SOUDAINE (A_narrow -> A_wide)")
    print("-" * 50)
    print(f"   -> K = 0           : {res_exp['K=0']:>9.2f} Pa")
    print(f"   -> internal = yes  : {res_exp['internal=yes']:>9.2f} Pa")
    print(f"   -> K = 1000        : {res_exp['K=1000']:>9.2f} Pa")

    print("\n2. CONTRACTION SOUDAINE (A_wide -> A_narrow)")
    print("-" * 50)
    print(f"   -> K = 0           : {res_con['K=0']:>9.2f} Pa")
    print(f"   -> internal = yes  : {res_con['internal=yes']:>9.2f} Pa")
    print(f"   -> K = 1000        : {res_con['K=1000']:>9.2f} Pa")
    print("="*50 + "\n")