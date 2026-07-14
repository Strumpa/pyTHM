import numpy as np
from pyTHM.Solver.main import pyTHM_solver

# ==========================================
# Paramètres issus de la Table II du papier
# ==========================================
I_z = 28                     # 20 noeuds en bas + 8 noeuds en haut
L_tot = 4.53e-3              # 4.53 mm : extrêmement court pour annuler gravité et friction
flow_rate = 9.506            # kg/s
P_out = 7.18e6               # 7.18 MPa
T_in = 550.0                 # ~277°C (Liquide sous-refroidi pour rester en monophasique)

# --- Profil des sections de passage (Expansion) ---
A_narrow = 0.010334
A_wide = 0.10334
# Les 20 premiers noeuds sont étroits, les 8 derniers sont larges
acools_exp = np.array([A_narrow] * 20 + [A_wide] * 8)

# --- Paramètres géométriques déduits ---
# Le diamètre hydraulique importe peu vu la très faible longueur, on met la relation classique
dhs_exp = np.sqrt(4 * acools_exp / np.pi)
phs_exp = np.ones(I_z)
porosities = np.ones(I_z)

# --- Profils des pertes de charge singulières ---
# La transition se fait entre la maille 20 (index 19) et la maille 21 (index 20)
# Donc la "face" de changement de section est la face d'index 20 (sur les 29 faces au total)
kexp_profile = np.zeros(I_z + 1)
kcon_profile = np.zeros(I_z + 1)
rsin_profile = np.zeros(I_z + 1)
#kexp_profile[20] = (A_wide/A_narrow)**2 * (1-A_narrow/A_wide)**2  # Formule de perte de charge pour expansion brusque
#kexp_profile[20] = 1000  

# /!\ OPTION 1 : Cas réversible (K=0) du papier (Delta P attendu = -550 Pa)
# On laisse kexp_profile à 0.0 partout.

# /!\ OPTION 2 : Cas irréversible (K=1000) du papier (Delta P attendu = +5017 Pa)
# Décommentez la ligne ci-dessous pour tester l'impact du K-factor :
# kexp_profile[20] = 1000.0

# ==========================================
# Lancement du solveur pyTHM
# ==========================================
THsolve = pyTHM_solver(
    case_name="TableII_Expansion",
    water_rod=False,         # Résolution canal actif seul
    canal_type="square",
    canal_radius=0.065,      # Rayons fictifs (inutilisés vu que Powtot=0)
    fuel_radius=0.004,
    gap_radius=0.0041,
    clad_radius=0.0047,
    fuel_rod_length=L_tot,
    tInlet=T_in,
    pOutlet=P_out,
    qFlow=flow_rate,
    Powtot=0.0,              # Puissance nulle
    axial_p_form=np.ones(I_z),  
    fraction_pow_fuel=1.0,
    k_fuel=3.0, H_gap=10000.0, k_clad=15.0,
    I_z=I_z, I_f=8, I_c=3,
    plot_at_z=[],
    solveConduction=False,   # Pas de conduction thermique nécessaire
    dt=0, t_tot=0,
    frfaccorel='null',       # Désactivation de la friction pariétale pour coller au papier
    P2Pcorel='base',
    voidFractionCorrel='EPRIvoidModel',
    numericalMethod="FVM",
    porosities=porosities,
    acools=acools_exp,
    dhs=dhs_exp,
    phs=phs_exp,
    kexp_profile=kexp_profile,
    kcon_profile=kcon_profile,
    rsin_profile=rsin_profile
)

# ==========================================
# Analyse des résultats
# ==========================================
P_array = THsolve.convection_sol.P[-1]
# La perte de charge totale = Pression entrée - Pression sortie
Delta_P = P_array[0] - P_array[-1]

print("\n" + "="*50)
print(f"RÉSULTAT DU TEST D'EXPANSION")
print("="*50)
print(f"Pression d'entrée : {P_array[0]:.2f} Pa")
print(f"Pression de sortie: {P_array[-1]:.2f} Pa")
print(f"Delta P calculé   : {Delta_P:.2f} Pa")
print("Valeurs attendues selon Table III du papier PARCS/PATHS :")
print(" -> Si K=0    : ~ -550 Pa (Regain de pression Bernoulli)")
print(" -> Si K=1000 : ~ +5017 Pa (Perte de charge massive)")