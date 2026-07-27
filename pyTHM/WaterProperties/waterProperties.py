# This file contains the classes and functions to create fields and give the values of the variables in the drift flux model reoslution file
# This class models the thermohydraulic behavior of a fluid flow in a system divided into multiple cells. Each cell is characterized by properties such as velocity (U), pressure (P), enthalpy (H), void fraction, and various geometric parameters. 
#The class employs different correlations to compute key thermophysical properties, and performs iterative updates to simulate two-phase flow dynamics.
# The file is used in the BWR/driftFluxModel/thermalHydraulicsTransitoire/THM_convection.py file
#Authors: Clément Huet
#Date: 2024-09-16

import numpy as np
from iapws import IAPWS97
from scipy.interpolate import RectBivariateSpline
from pyTHM.Lissage.smooth import if_lisse, max_lisse, min_lisse

class FastIAPWS:
    """
    Générateur de Look-Up Tables 1D (Saturation) et 2D (Sous-refroidi).
    """
    def __init__(self):
        print("--- Initialisation des tables thermodynamiques (IAPWS97) ---")
        # --- 1. TABLES DE SATURATION 1D (Pression uniquement) ---
        self.P_arr = np.linspace(6.5, 7.5, 200) # MPa
        
        self.Tsat, self.hl, self.hg = np.zeros(200), np.zeros(200), np.zeros(200)
        self.rhol, self.rhog = np.zeros(200), np.zeros(200)
        self.mul, self.mug = np.zeros(200), np.zeros(200)
        self.sigma = np.zeros(200)
        self.cpl, self.cpg = np.zeros(200), np.zeros(200)
        self.kl, self.kg = np.zeros(200), np.zeros(200)
        
        for i, p in enumerate(self.P_arr):
            liq = IAPWS97(P=p, x=0)
            vap = IAPWS97(P=p, x=1)
            self.Tsat[i] = liq.T
            self.hl[i] = liq.h
            self.hg[i] = vap.h
            self.rhol[i] = liq.rho
            self.rhog[i] = vap.rho
            self.mul[i] = liq.mu
            self.mug[i] = vap.mu
            self.sigma[i] = liq.sigma
            self.cpl[i] = liq.cp
            self.cpg[i] = vap.cp
            self.kl[i] = liq.k
            self.kg[i] = vap.k
            
        # --- 2. TABLES SOUS-REFROIDIES 2D (Pression ET Enthalpie) ---
        print("--- Génération des surfaces 2D (P, H)... ---")
        self.P_sub_arr = np.linspace(6.5, 7.5, 50)     # Grille de Pression
        self.H_sub_arr = np.linspace(300.0, 1600.0, 200) # Grille d'Enthalpie
        
        rhol_sub_2d = np.zeros((50, 200))
        T_sub_2d = np.zeros((50, 200))
        
        for i, p in enumerate(self.P_sub_arr):
            for j, h_val in enumerate(self.H_sub_arr):
                state = IAPWS97(P=p, h=h_val)
                rhol_sub_2d[i, j] = state.rho
                T_sub_2d[i, j] = state.T
                
        # Création des fonctions d'interpolation 2D ultra-rapides en C
        self.rhol_sub_spline = RectBivariateSpline(self.P_sub_arr, self.H_sub_arr, rhol_sub_2d)
        self.T_sub_spline = RectBivariateSpline(self.P_sub_arr, self.H_sub_arr, T_sub_2d)
            
        print("--- Tables IAPWS générées avec succès ! ---")

    # Méthodes d'accès 1D
    def get_Tsat(self, P): return np.interp(P, self.P_arr, self.Tsat)
    def get_hl(self, P): return np.interp(P, self.P_arr, self.hl)
    def get_hg(self, P): return np.interp(P, self.P_arr, self.hg)
    def get_rhol(self, P): return np.interp(P, self.P_arr, self.rhol)
    def get_rhog(self, P): return np.interp(P, self.P_arr, self.rhog)
    def get_mul(self, P): return np.interp(P, self.P_arr, self.mul)
    def get_mug(self, P): return np.interp(P, self.P_arr, self.mug)
    def get_sigma(self, P): return np.interp(P, self.P_arr, self.sigma)
    def get_cpl(self, P): return np.interp(P, self.P_arr, self.cpl)
    def get_cpg(self, P): return np.interp(P, self.P_arr, self.cpg)
    def get_kl(self, P): return np.interp(P, self.P_arr, self.kl)
    def get_kg(self, P): return np.interp(P, self.P_arr, self.kg)
    
    # Méthodes d'accès 2D (grid=False force l'évaluation point par point)
    def get_sub_rhol(self, P, H): 
        # On utilise np.ravel()[0] pour être certain de renvoyer un scalaire pur
        return np.ravel(self.rhol_sub_spline(P, H, grid=False))[0]
        
    def get_sub_T(self, P, H): 
        return np.ravel(self.T_sub_spline(P, H, grid=False))[0]

FAST_IAPWS = FastIAPWS()

class statesVariables():

    """
    Attributes:
    - nCells (int): Number of cells in the system.
    - U (array): Velocity values for each cell.
    - P (array): Pressure values for each cell.
    - H (array): Enthalpy values for each cell.
    - voidFraction (array): Initial void fraction values for each cell.
    - voidFractionCorrel (str): Correlation model used for void fraction calculations (e.g., 'modBestion', 'HEM1').
    - frfaccorel (str): Correlation model used for friction factor calculations (e.g., 'base', 'blasius', 'Churchill').
    - P2Pcorel (str): Correlation model used for two-phase pressure multiplier (e.g., 'base', 'HEM1').
    - D_h (float): Hydraulic diameter of the system.
    - flowArea (float): Flow area of the system.
    - DV (float): Differential volume between cells.
    - g (float): Gravitational acceleration (9.81 m/s²).
    - K_loss (float): Loss coefficient in the system.
    - Dz (float): Distance between cells.
    - Poro (float): Porosity of the system.

    Methods:
    - createFields(): Initializes temporary arrays for cell-specific properties such as densities, void fractions, friction factors, and geometric areas. Uses specific methods to populate these arrays.
    - updateFields(): Updates key properties (e.g., void fraction, densities) based on the selected void fraction correlation model. Calls corresponding methods for each model.
    - modBestion(): Updates void fraction and related properties using the "modBestion" correlation with iterative convergence.
    - HEM1(): Updates void fraction and related properties using the "HEM1" correlation with iterative convergence.
    - GEramp(): Updates void fraction and related properties using the "GEramp" correlation with iterative convergence.
    - EPRIvoidModel(): Updates void fraction and related properties using the "EPRIvoidModel" correlation with iterative convergence.
    - getDensity(i): Retrieves the liquid and vapor densities for a given cell using IAPWS97 thermodynamic models.
    - getQuality(i): Calculates the quality (phase fraction) based on enthalpy values.
    - getVoidFraction(i): Computes void fraction using different correlations, depending on the selected model.
    - getVgj(i): Calculates the drift velocity for a given cell based on the selected void fraction correlation.
    - getC0(i): Computes the slip ratio for a given cell based on the selected correlation model.
    - getFrictionFactor(i): Calculates the friction factor based on Reynolds number and the selected friction factor correlation.
    - getPhi2Phi(i): Computes the two-phase pressure multiplier for a given cell.
    - getAreas(i): Calculates the positive and negative flow areas for a given cell.
    - getPhasesEnthalpy(i): Retrieves the enthalpy values for the liquid and vapor phases at a given pressure.
    - getReynoldsNumber(i): Computes the Reynolds number for flow in a given cell.
    """

    def __init__(self, U, P, H, voidFraction, clad_radius, pin_pitch, pitch, D_h, areaMatrix, poro, DV, voidFractionCorrel, frfaccorel, P2Pcorel, Dz, q__, phs, qFlow, rf, rw, kexp, kcon, rsin):
        
        self.nCells = len(U)
        self.U = U
        self.P = P
        self.H = H
        self.voidFraction = voidFraction
        self.voidFractionCorrel = voidFractionCorrel
        self.frfaccorel = frfaccorel
        self.P2Pcorel = P2Pcorel
        self.g = 9.81
        self.D_h = D_h
        self.areaMatrix = areaMatrix
        self.poro = poro
        self.K_loss = 0#0.1
        self.Dz = Dz
        self.DV = DV
        self.q__ = q__
        self.phs = phs
        self.qFlow = qFlow
        self.rf = rf
        self.rw = rw
        self.height = self.Dz * self.nCells
        self.kexp = kexp
        self.kcon = kcon
        self.rsin = rsin
        self.clad_radius = clad_radius
        self.pin_pitch = pin_pitch
        self.pitch = pitch
    #Create the array to store the temporary values of the variables
    def createFields(self):
        self.areaMatrixTEMP = np.ones(self.nCells)
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionTEMP = self.voidFraction
        self.xThTEMP = np.ones(self.nCells)
        for i in range(self.nCells):
            self.xThTEMP[i] = self.getQuality(i)
            self.areaMatrixTEMP[i] = self.areaMatrix[i]
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)

    #Update the values of the variables using the selected void fraction correlation model
    def updateFields(self):

        self.xThTEMP = np.ones(self.nCells)
        for i in range(self.nCells):
            self.xThTEMP[i] = self.getQuality(i)
        
        if self.voidFractionCorrel == 'modBestion':
            self.modBestion()

        elif self.voidFractionCorrel == 'HEM1':
            self.HEM1()

        elif self.voidFractionCorrel == 'GEramp':
            self.GEramp()

        elif self.voidFractionCorrel == 'EPRIvoidModel':
            self.EPRIvoidModel()
        
        elif self.voidFractionCorrel == 'Hibiki_Al-Saif':
            self.Hibiki_Al_Saif()

        elif self.voidFractionCorrel == 'Ozaki':
            self.Ozaki()
        else:
            raise ValueError('Invalid void fraction correlation model')

    #Update the values of the variables using the selected void fraction correlation model: modBestion
    def modBestion(self):
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        self.Ul = np.ones(self.nCells)
        self.Ug = np.ones(self.nCells)
        self.Rel = np.ones(self.nCells)
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            voidFractionNew = self.getVoidFraction(i)
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)
            self.Ul[i] = self.getUl(i)
            self.Ug[i] = self.getUg(i)
            self.Rel[i] = self.getReynoldsNumberLiquid(i)
    
    #Update the values of the variables using the selected void fraction correlation model: HEM1
    def HEM1(self):
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        self.Ul = np.ones(self.nCells)
        self.Ug = np.ones(self.nCells)
        self.Rel = np.ones(self.nCells)
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            voidFractionNew = self.getVoidFraction(i)
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)
            self.Ul[i] = self.getUl(i)
            self.Ug[i] = self.getUg(i)
            self.Rel[i] = self.getReynoldsNumberLiquid(i)

    #Update the values of the variables using the selected void fraction correlation model: GEramp
    def GEramp(self):
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        self.Ul = np.ones(self.nCells)
        self.Ug = np.ones(self.nCells)
        self.Rel = np.ones(self.nCells)
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            for j in range(1000):
                voidFractionNew = self.getVoidFraction(i)
                if np.linalg.norm(voidFractionNew - self.voidFractionTEMP[i]) < 1e-3:
                    self.voidFractionTEMP[i] = voidFractionNew
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
                    break
                elif j == 999:
                    raise ValueError('Convergence in update fields not reached')
                    break
                else:
                    self.voidFractionTEMP[i] = voidFractionNew
                    if self.voidFractionTEMP[i] < 0.0:
                        self.voidFractionTEMP[i] = 0.0
                    elif self.voidFractionTEMP[i] > 0.999:
                        self.voidFractionTEMP[i] = 0.999
                        
                    if self.xThTEMP[i] < 0.0:
                        self.xThTEMP[i] = 0.0
                    elif self.xThTEMP[i] > 1.0:
                        self.xThTEMP[i] = 1.0
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)

            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)
            self.Ul[i] = self.getUl(i)
            self.Ug[i] = self.getUg(i)
            self.Rel[i] = self.getReynoldsNumberLiquid(i)

    #Update the values of the variables using the selected void fraction correlation model: EPRIvoidModel
    def EPRIvoidModel(self):
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        self.Ul = np.ones(self.nCells)
        self.Ug = np.ones(self.nCells)
        self.Rel = np.ones(self.nCells)
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            for j in range(1000):
                voidFractionNew = self.getVoidFraction(i)
                if np.linalg.norm(voidFractionNew - self.voidFractionTEMP[i]) < 1e-3:
                    self.voidFractionTEMP[i] = voidFractionNew
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
                    break
                elif j == 999:
                    raise ValueError('Convergence in update fields not reached')
                    break
                else:
                    self.voidFractionTEMP[i] = voidFractionNew
                    if self.voidFractionTEMP[i] < 0.0:
                        self.voidFractionTEMP[i] = 0.0
                    elif self.voidFractionTEMP[i] > 0.999:
                        self.voidFractionTEMP[i] = 0.999
                        
                    if self.xThTEMP[i] < 0.0:
                        self.xThTEMP[i] = 0.0
                    elif self.xThTEMP[i] > 1.0:
                        self.xThTEMP[i] = 1.0
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)

            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)
            self.Ul[i] = self.getUl(i)
            self.Ug[i] = self.getUg(i)
            self.Rel[i] = self.getReynoldsNumberLiquidD5(i)

    def Hibiki_Al_Saif(self):
        # 1. Initialisation des tableaux
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        self.Ul = np.ones(self.nCells)
        self.Ug = np.ones(self.nCells)
        self.Rel = np.ones(self.nCells)
        
        for i in range(self.nCells):
            # 2. Extraction des densités pures (briques de base)
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            
            # 3. Calcul de C0 et Vgj (Ils ne dépendent plus que de G, x et rho !)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            
            # 4. Déduction de la fraction de vide (via l'équation DFM standard)
            voidFractionNew = self.getVoidFraction(i)
            
            # Sécurité numérique sur le taux de vide
            if voidFractionNew < 0.0:
                voidFractionNew = 0.0
            elif voidFractionNew > 0.999:
                voidFractionNew = 0.999
            self.voidFractionTEMP[i] = voidFractionNew
            
            # 5. Calcul de la densité du mélange maintenant qu'on a le taux de vide
            self.rhoTEMP[i] = self.getDensity(i)[2]
            
            # 6. Déduction des produits dérivés (Vitesses des phases)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.Ul[i] = self.getUl(i)
            self.Ug[i] = self.getUg(i)
            
            # 7. Clôture des paramètres hydrauliques
            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)
            self.Rel[i] = self.getReynoldsNumberLiquidD5(i)

    #Update the values of the variables using the selected void fraction correlation model: Ozaki
    def Ozaki(self):
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        self.Ul = np.ones(self.nCells)
        self.Ug = np.ones(self.nCells)
        self.Rel = np.ones(self.nCells)
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            omega = 0.5 # Relaxation
            for j in range(1000):
                voidFractionNew = self.getVoidFraction(i)
                if np.linalg.norm(voidFractionNew - self.voidFractionTEMP[i]) < 1e-3:
                    self.voidFractionTEMP[i] = voidFractionNew
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
                    break
                elif j == 999:
                    raise ValueError('Convergence in update fields not reached for Ozaki')
                else:
                    self.voidFractionTEMP[i] = omega * voidFractionNew + (1.0 - omega) * self.voidFractionTEMP[i]
                    
                    # Bornes physiques strictes
                    if self.voidFractionTEMP[i] < 0.0:
                        self.voidFractionTEMP[i] = 0.0
                    elif self.voidFractionTEMP[i] > 0.999:
                        self.voidFractionTEMP[i] = 0.999

                    if self.xThTEMP[i] < 0.0:
                        self.xThTEMP[i] = 0.0
                    elif self.xThTEMP[i] > 1.0:
                        self.xThTEMP[i] = 1.0
                        
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)

            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)
            self.Ul[i] = self.getUl(i)
            self.Ug[i] = self.getUg(i)
            self.Rel[i] = self.getReynoldsNumberLiquid(i)

    #Get the density of the liquid and vapor phases for a given cell
    def getDensity(self, i):
        P = self.P[i]
        H = self.H[i]
        hl, hg = self.getPhasesEnthalpy(i)
        H_liq_eval = min_lisse(H*(10**(-3)), hl, 0.5)
        rho_g = FAST_IAPWS.get_rhog(P*(10**(-6)))
        rho_l = FAST_IAPWS.get_sub_rhol(P*(10**-6), H_liq_eval)
        rho = rho_l * (1 - self.voidFractionTEMP[i]) + rho_g * self.voidFractionTEMP[i]
        return rho_l, rho_g, rho
    
    #Get the quality  based on enthalpy values
    def getQuality(self, i):
        correl = "simple"
        if correl == 'simple':
            hl, hg = self.getPhasesEnthalpy(i)
            H = self.H[i] * 0.001
            x_raw = (H - hl)/(hg - hl)
            x_borne_sup = min_lisse(1.0, x_raw, 1e-4)
            x_final = max_lisse(0.0, x_borne_sup, 1e-4)
            return x_final
        
        elif correl == 'EPRI':
            epriCorrel = '1'
            Xs = 0.05
            Xh = Xs / 2
            hl, hg = self.getPhasesEnthalpy(i)
            hl = hl
            hg = hg
            H = self.H[i]
            p = self.P[i]
            xeq = (hl - H*0.001) / (hl - hg)
            
            if epriCorrel == "1":
                if xeq >= Xs:
                    QUALITY = xeq
                else:
                    # --- 1. CORRECTION DES DENSITÉS ---
                    # On calcule les densités exactes via IAPWS pour éviter l'AttributeError
                    # et s'affranchir de self.rholTEMP qui n'est pas encore mis à jour.
                    p_mpa = p * 1e-6
                    H_liq_eval = min_lisse(H * 1e-3, hl, 0.5)
                    rhol = FAST_IAPWS.get_sub_rhol(p_mpa, H_liq_eval)
                    rhog = FAST_IAPWS.get_rhog(p_mpa)
                    
                    u = self.U[i]
                    
                    # --- 2. CORRECTION DES PROPRIÉTÉS DU LIQUIDE ---
                    # Le modèle exige les propriétés du LIQUIDE (suffixe 'l') 
                    # et non de la vapeur (suffixe 'g') !
                    muf = FAST_IAPWS.get_mul(p_mpa)
                    Re = rhol * abs(u) * self.D_h[i] / muf

                    Cpf = FAST_IAPWS.get_cpl(p_mpa)
                    k_f = FAST_IAPWS.get_kl(p_mpa)
                    Pr = Cpf * muf / k_f

                    # --- 3. CORRECTION DES INDEX HORS-LIMITES ---
                    idx_q = min(i, len(self.q__) - 1)
                    idx_dv = min(i, len(self.DV) - 1)
                    qdp = self.q__[idx_q] * self.DV[idx_dv] / (2 * np.pi * self.rw * self.height)

                    # Calculate heat transfer coefficients
                    hb = np.exp(p / 4.35e6) / (22.7)**2 * 1000.0  # W/(m^2·K)
                    Chn = 0.2 / 4.0 * self.D_h[i] / self.rf
                    hhn = Chn * Re**0.662 * Pr * k_f / (self.D_h[i])
                    Cdb = (0.033 * self.areaMatrix[i] / (self.areaMatrix[i]/self.poro[i]) + 0.013)
                    hdb = Cdb * Re**0.8 * Pr**0.4 * k_f / (self.D_h[i])

                    # Intermediate calculations
                    tmp1 = 4.0 * hb * (hdb + hhn)**2
                    tmp2 = 2.0 * hdb**2 * (hhn + hdb / 2.0) + 8.0 * qdp * hb * (hdb + hhn)
                    tmp3 = qdp * (4.0 * hb * qdp + hdb**2)

                    # Calculate characteristic quality xd
                    delta_h = hg - hl
                    xd = -Cpf / (delta_h) * (
                        (-tmp2 + np.sqrt(tmp2**2 - 4.0 * tmp1 * tmp3)) / (2.0 * tmp1)
                    )

                    # Determine quality based on xeq and xd
                    if xeq <= 0.0:
                        if xeq <= -xd:
                            QUALITY = 0.0       
                        else:
                            tmp1 = 1 + xeq / xd
                            tmp2 = tmp1**2
                            QUALITY = xd * tmp2 * (0.1 + 0.087 * tmp1 + 0.05 * tmp2)
                    elif Xh > xd:
                        if xeq >= 2 * xd:
                            QUALITY = xeq
                        else:
                            tmp1 = xeq / xd
                            QUALITY = xd * (0.237 + tmp1 * (0.661 + tmp1 * (0.153 + tmp1 * (-0.01725 - tmp1 * 0.0020625))))
                    else:
                        tmp1 = xeq / Xh
                        tmp2 = xd / Xh
                        tmp3 = 0.237 * tmp2
                        QUALITY = Xh * (tmp3 + tmp1 * (0.661 + tmp1 * (0.5085 - 0.3555 * tmp2 + tmp1 * (tmp3 - 0.25425 + tmp1 * (0.042375 - 0.0444375 * tmp2)))))

                if QUALITY >= 1.0:
                    return 0.99
                else:
                    return QUALITY

    #Get the void fraction for a given cell using different correlations for drift flulx model
    def getVoidFraction(self, i):
        correl = 'paths'
        if correl == 'simple':
            x_th = self.xThTEMP[i]
            rho_l = self.rholTEMP[i]
            rho_g = self.rhogTEMP[i]
            if x_th == 0:
                return 0
            elif x_th == 1:
                return 0.99
            else:
                return (x_th * rho_l)/(x_th * rho_l + (1 - x_th) * rho_g)
        elif correl == 'paths':
            x_th = self.xThTEMP[i]
            rho_l = self.rholTEMP[i]
            rho_g = self.rhogTEMP[i]
            rho_m = self.rhoTEMP[i]
            u = abs(self.U[i])
            V_gj = self.VgjTEMP[i]
            C0 = self.C0TEMP[i]
            if x_th == 0:
                return 0
            elif x_th == 1:
                return 0.99
            else:
                return x_th / (C0 * (x_th + (rho_g / rho_l) * (1 - x_th)) + (rho_g * V_gj) / (rho_m * u))
    
    #Get the drift velocity for a given cell based on the selected void fraction correlation
    def getVgj(self, i):
        if self.voidFractionCorrel == 'GEramp':
            if self.rhogTEMP[i] == 0:
                return 0
            if self.rholTEMP[i] == 0:
                return 0
            
            sigma = FAST_IAPWS.get_sigma(self.P[i]*(10**(-6)))
            if sigma == 0:
                return 0
            
            Vgj0 = ((self.g * sigma * (self.rholTEMP[i] - self.rhogTEMP[i]) / self.rholTEMP[i]**2)**0.25)

            if self.voidFractionTEMP[i] <= 0.65:
                return 2.9 * Vgj0
            elif self.voidFractionTEMP[i] > 0.65:
                return (2.9/0.35)*(1-self.voidFractionTEMP[i]) * Vgj0
        
        if self.voidFractionCorrel == 'modBestion':
            if self.rhogTEMP[i] == 0:
                return 0
            if self.rholTEMP[i] == 0:
                return 0
            return 0.188 * np.sqrt(((self.rholTEMP[i] - self.rhogTEMP[i]) * self.g * self.D_h[i] ) / self.rhogTEMP[i] )
        
        if self.voidFractionCorrel == 'EPRIvoidModel':
            if self.rhogTEMP[i] == 0:
                return 0
            if self.rholTEMP[i] == 0:
                return 0
            sigma = FAST_IAPWS.get_sigma(self.P[i]*(10**(-6)))
            Vgj = (np.sqrt(2)*(self.g * sigma * (self.rholTEMP[i] - self.rhogTEMP[i]) / self.rholTEMP[i]**2)**0.25) * (1 + self.voidFractionTEMP[i])**(3/2)
            return Vgj
        
        if self.voidFractionCorrel == 'HEM1':
            return 0

        # Issu des papiers de Hibiki et al. et Al-Saif et al.: 
        # Drift-flux model for upward dispersed two-phase flows in vertical medium-to-large round tubes
        # Accurate modeling of annular gas-water flow across diverse inclination angles using an advanced drift-flux correlation
        if self.voidFractionCorrel == 'Hibiki_Al-Saif':
            if self.rhogTEMP[i] == 0:
                return 0    
            if self.rholTEMP[i] == 0:
                return 0
            P = self.P[i]
            H = self.H[i]
            hl, hg = self.getPhasesEnthalpy(i)
            T = FAST_IAPWS.get_sub_T(P*(10**(-6)), H*10**(-3))
            Tsat = FAST_IAPWS.get_Tsat(P*(10**(-6)))
            DTSUB = self.getDTSUB(i)
            H_kJ = H*10**(-3)
            G = self.rhoTEMP[i]*abs(self.U[i])
            x = self.xThTEMP[i]
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            D_h = self.D_h[i]
            mul = FAST_IAPWS.get_mul(P*(10**(-6)))
            Re = max(1e-4, G*D_h/mul)
            sigma = FAST_IAPWS.get_sigma(P*(10**(-6)))
            C_inf = 1.393 - 0.015*np.log(Re)
            eps_tran = (1 + 4*rho_g/rho_l)*np.sqrt(rho_l/rho_g)/(C_inf*(C_inf-1)) - 4*rho_g/rho_l
            La = (sigma/((rho_l-rho_g)*self.g))**0.5
            N_mul = mul/(rho_l*sigma*La)**(0.5)
            N_rho = rho_g/rho_l
            Dh_ratio = D_h/La
            if N_mul <= 2*10**(-3):
                if Dh_ratio <= 30:
                    Vgj_KI_plus = 0.019*Dh_ratio**(0.809)*N_rho**(-0.157)*N_mul**(-0.562)
                else:
                    Vgj_KI_plus = 0.030*N_rho**(-0.157)*N_mul**(-0.562)
            else:
                if Dh_ratio >= 30:
                    Vgj_KI_plus = 0.92*N_rho**(-0.157)
                else:
                    Vgj_KI_plus = 0
            plus = ((rho_l-rho_g)*self.g*sigma/rho_l**2)**0.25
            jg_plus = (G*x)/(rho_g*plus)
            jl_plus = (G*(1-x))/(rho_l*plus)
            j_plus = jg_plus + jl_plus
            denomin_KI = self.getC0(i)*j_plus + Vgj_KI_plus
            alpha_KI = jg_plus / denomin_KI if denomin_KI > 1e-5 else 0.0
            Vgj_plus = np.sqrt(2)*(1-alpha_KI)**(1.75)*np.exp(-60.63*alpha_KI**(2.367)) + Vgj_KI_plus*(1-np.exp(-60.63*alpha_KI**(2.367)))
            Vgj_bubbly = Vgj_plus * plus
            Vgj_slug = 0.193*(self.g *  sigma *(rho_l-rho_g) / rho_l**2)**0.25
            Vgj_diphasique = if_lisse(self.voidFractionTEMP[i], eps_tran, 0.05, Vgj_slug, Vgj_bubbly)
            Vgj_final = if_lisse(T, Tsat-DTSUB, 0.5, Vgj_diphasique, 0.0)
            return Vgj_final

        #Issu du papier de Ozaki et al: Development of drift-flux model based on 8 × 8 BWR rod bundle geometry experimentsunder prototypic temperature and pressure conditions
        if self.voidFractionCorrel == 'Ozaki':
            if self.rhogTEMP[i] == 0:
                return 0
            if self.rholTEMP[i] == 0:
                return 0
            P = self.P[i]
            G = self.rhoTEMP[i]*abs(self.U[i])
            x = self.xThTEMP[i]
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            mul = FAST_IAPWS.get_mul(P*(10**(-6)))
            sigma = FAST_IAPWS.get_sigma(P*(10**(-6)))
            eps = self.voidFractionTEMP[i]
            La = (sigma/((rho_l-rho_g)*self.g))**0.5
            N_mul = mul/(rho_l*sigma*La)**(0.5)
            N_rho = rho_g/rho_l
            Dh_ratio = self.pitch/La
            plus = ((rho_l-rho_g)*self.g*sigma/rho_l**2)**0.25
            jg_plus = (G*x)/(rho_g*plus)
            Vgj_plus_B = np.sqrt(2) * (1 - eps)**1.75
            if N_mul <= 2.25*10**(-3):
                if Dh_ratio <= 30:
                    Vgj_plus_P = 0.019 * Dh_ratio**(0.809) * N_rho**(-0.157) * N_mul**(-0.562)
                else:
                    Vgj_plus_P = 0.030 * N_rho**(-0.157) * N_mul**(-0.562)
            else:
                if Dh_ratio >= 30:
                    Vgj_plus_P = 0.92 * N_rho**(-0.157)
                else:
                    Vgj_plus_P = 0
            Vgj_plus = Vgj_plus_B * np.exp(-1.39 * jg_plus) + Vgj_plus_P * (1 - np.exp(-1.39 * jg_plus))
            Vgj_final = Vgj_plus * plus
            return Vgj_final
            
    #Get the distribution parameter for a given cell based on the selected correlation model
    def getC0(self, i):
        if self.voidFractionCorrel == 'GEramp':
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            if rho_g == 0:
                return 0
            if rho_l == 0:
                return 0
            if self.voidFractionTEMP[i] <= 0.65:
                return 1.1
            elif self.voidFractionTEMP[i] > 0.65:
                return 1 + (0.1/0.35)*(1-self.voidFractionTEMP[i])
        
        if self.voidFractionCorrel == 'modBestion':
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            if rho_g == 0:
                return 0
            if rho_l == 0:
                return 0
            return 1.2 - 0.2*np.sqrt(rho_g / rho_l)
        
        if self.voidFractionCorrel == 'EPRIvoidModel':
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            Pc = 22060000
            P = self.P[i]
            Re = self.getReynoldsNumber(i)
            C1 = (4 * Pc**2)/(P*(Pc - P))
            k1 = min(0.8, 1/(1 + np.exp(-Re /60000)))
            k0 = k1 + (1-k1) * (rho_g / rho_l)**2
            r = (1+1.57*(rho_g/rho_l))/(1-k1)
            C0 = (((k0 + (1 - k0) * (self.voidFractionTEMP[i]**r))**(-1)) * ((1 - np.exp(-C1 * self.voidFractionTEMP[i]))/(1 - np.exp(-C1))))
            return C0

        if self.voidFractionCorrel == 'HEM1':
            return 1
        
        if self.voidFractionCorrel == 'Hibiki_Al-Saif':
            G = self.rhoTEMP[i] * abs(self.U[i])
            x = self.xThTEMP[i] 
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            N_rho = rho_g/rho_l
            P = self.P[i]
            sigma = FAST_IAPWS.get_sigma(P*(10**(-6)))
            plus = ((rho_l-rho_g)*self.g*sigma/rho_l**2)**0.25
            jg_plus = (G*x)/(rho_g*plus)
            jl_plus = (G*(1-x))/(rho_l*plus)
            j_plus = jg_plus + jl_plus
            alpha_CT = jg_plus/((1.2-0.2*N_rho**0.5)*j_plus+np.sqrt(2))
            hl, hg = self.getPhasesEnthalpy(i)
            H = self.H[i]*10**(-3)
            T = FAST_IAPWS.get_sub_T(P*(10**(-6)), H)
            Tsat = FAST_IAPWS.get_Tsat(P*(10**(-6)))
            DTSUB = self.getDTSUB(i)
            C0_liq = 1.0
            C0_sat = 1.2 - 0.2 * N_rho**0.5
            C0_sub = C0_sat * (1 - np.exp(-18*alpha_CT))
            C0_trans = if_lisse(T, Tsat - DTSUB, 0.5, C0_sub, C0_liq)
            C0_final = if_lisse(H, hl, 5.0, C0_sat, C0_trans)
            return C0_final

        #Issu du papier de Ozaki et al: Development of drift-flux model based on 8 × 8 BWR rod bundle geometry experimentsunder prototypic temperature and pressure conditions
        if self.voidFractionCorrel == 'Ozaki':
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            if rho_g == 0:
                return 0
            if rho_l == 0:
                return 0
            ratio = self.clad_radius * 2.0 / self.pin_pitch
            A = 1.015 + 0.05 * ratio
            B = 0.015 + 0.05 * ratio
            ratio_table = np.array([0.3, 0.5, 0.7]) # pas idéal, il faut trouver le papier de Julia et al. pour une expression explicite de C et D : Julia JE, Hibiki T, Ishii M, Yun BJ, Park GC. Drift-fluxmodel in a sub-channel of rod bundle geometry
            C_table = np.array([26.3, 21.2, 34.1])
            D_table = np.array([0.780, 0.762, 0.925])
            idx = np.argmin(np.abs(ratio_table - ratio))
            C = C_table[idx]
            D = D_table[idx]
            eps = max(0.0,self.voidFractionTEMP[i])
            eps_H = 0.2
            eps_L = 0.1
            C0_H = 1.1 - 0.1 * np.sqrt(rho_g / rho_l)
            if eps < 1e-6:
                C0_L = 0.0
            else:
                C0_L = (A - B * np.sqrt(rho_g / rho_l)) * (1 - np.exp(-C * eps**D))
            if eps >= eps_H:
                return C0_H
            if eps <= eps_L:
                return C0_L
            else:
                f = (eps_H - eps) / (eps_H - eps_L)
                return C0_L*f + C0_H*(1-f)
            
    #Get the drift velocity for a given cell based on the selected void fraction correlation
    def getVgj_prime(self, i):
        U = self.U[i]
        C0 = self.C0TEMP[i]
        Vgj = self.VgjTEMP[i]
        Vgj_prime = Vgj + (C0-1) * U
        return Vgj_prime
    
    def getHfg(self, i):
        hg = FAST_IAPWS.get_hg(self.P[i]*(10**(-6)))
        hl = FAST_IAPWS.get_hl(self.P[i]*(10**(-6)))
        return (hg - hl)
    
    def getFrictionFactor(self, i):
        U = self.U[i]
        P = self.P[i]
        Re = self.getReynoldsNumberLiquidD5(i)
        if (self.voidFractionTEMP[i]<0.002):
            return 0.316 * Re**(-0.25)
        if self.frfaccorel == 'base': #Validated
            return 0.003
        elif self.frfaccorel == "null": #Validated
            return 0
        elif self.frfaccorel == 'blasius': #Validated
            #return 0.316 * Re**(-0.25)
            return 0.079 * Re**(-0.25)
            #return 0.316 * Re**(-0.25)
        elif self.frfaccorel == 'Churchill': #Validated
            #old 0.4 When Ra increased pressure drop increase
            #Ra = 0.8 * (10**(-6)) #Roughness
            #Ra = 1.0 * (10**(-6)) #Roughness
            Ra = 0.4 * (10**(-6)) #Roughness
            R = Ra / self.D_h[i]
            frict=8*(((8.0/Re)**12)+((2.475*np.log(((7/Re)**0.9)+0.27*R))**16+(37530/Re)**16)**(-1.5))**(1/12)
            return frict
        elif self.frfaccorel == 'Churchill_notOK':
            Re = self.getReynoldsNumberLiquid(i)
            B = (37530/Re)**16
            A = (2.475*np.log(1/(((7/Re)**0.9)+0.27*(0.4/self.D_h[i]))))**16
            f  = 8*(((8/Re)**12) + (1/(A+B)**1.5))**(1/12)
            return f
        else:
            raise ValueError('Invalid friction factor correlation model')

    #Get the two-phase pressure multiplier for a given cell
    def getPhi2Phi(self, i):
        x_th = self.xThTEMP[i]
        rho_l = self.rholTEMP[i]
        rho_g = self.rhogTEMP[i]
        rho = self.rhoTEMP[i]
        P = self.P[i]
        epsilon = self.voidFractionTEMP[i]
        if epsilon <= 0.002:
            return 1
        if self.P2Pcorel == 'base': #Validated
            phi2phi = 1 + 3*epsilon
        elif self.P2Pcorel == 'lockhartMartinelli':
            return self.lockhartMartinelli(i)
        elif self.P2Pcorel == 'friedel':
            return self.friedel(i)
        elif self.P2Pcorel == 'HEM1': #Validated
            phi2phi = (rho/rho_l)*((rho_l/rho_g)*x_th + +1)
        elif self.P2Pcorel == 'HEM2': #Validated    
            m = FAST_IAPWS.get_mul(P*(10**(-6))) / FAST_IAPWS.get_mug(P*(10**(-6)))
            phi2phi = (rho/rho_l)*((m-1)*x_th + 1)*((rho_l/rho_g)*x_th + +1)**(0.25)
        elif self.P2Pcorel == 'MNmodel': #Validated
            phi2phi = (1.2 * (rho_l/rho_g -1)*x_th**(0.824) + 1)*(rho/rho_l)
        else:
            raise ValueError('Invalid two-phase pressure multiplier correlation model')
        return phi2phi
    def getPhi2Expansion(self, i):
        """
        Multiplicateur diphasique en expansion soudaine (Modèle de Romie)
        """
        x = self.xThTEMP[i]
        epsilon = self.voidFractionTEMP[i]
        rho_l = self.rholTEMP[i]
        rho_g = self.rhogTEMP[i]
        # Sécurité numérique : si on est en monophasique liquide pur
        if x <= 1e-5 or epsilon <= 1e-5:
            return 1.0
        if x >= 0.999 or epsilon >= 0.999:
            return (rho_l / rho_g)
        # Modèle de Romie
        phi2_exp = ((1 - x)**2) / (1 - epsilon) + (rho_l / rho_g) * (x**2 / epsilon)
        
        return phi2_exp

    def getPhi2Contraction(self, i):
        """
        Multiplicateur diphasique en contraction soudaine (Modèle de Chisholm)
        """
        x = self.xThTEMP[i]
        epsilon = self.voidFractionTEMP[i]
        rho_l = self.rholTEMP[i]
        rho_g = self.rhogTEMP[i]
        sigma_A = self.rsin[i]  # Ratio d'aire (A_petit / A_grand)
        
        # Sécurité numérique pour le monophasique ou si pas de contraction (sigma_A = 1)
        if x <= 1e-5 or epsilon <= 1e-5 or sigma_A >= 0.999:
            return 1.0
        
        if x >= 0.999 or epsilon >= 0.999 or sigma_A<=1e-5:
            return (rho_l / rho_g)
        # --- 1. Calcul du paramètre de Martinelli (X) ---
        mu_g = FAST_IAPWS.get_mug(self.P[i]*1e-6)
        mu_l = FAST_IAPWS.get_mul(self.P[i]*1e-6)
        X_LM = ((1 - x) / x)**0.9 * (rho_g / rho_l)**0.5 * (mu_g / mu_l)**0.1
        
        # --- 2. Calcul de K_o ---
        if X_LM >= 1.0:
            K_o = (1.0 + x * (rho_l / rho_g - 1.0))**0.5
        else:
            K_o = (rho_l / rho_g)**0.25
            
        # --- 3. Calcul de C_c (Équation 52) ---
        C_c = 1.0 / (0.639 * (1.0 - sigma_A)**0.5 + 1.0)
        
        # --- 4. Calcul de B ---
        # Numérateur de B
        num_B = (1.0 / K_o) * (1.0 / (sigma_A * C_c)**2 - 1.0) - (2.0 / (K_o * C_c * sigma_A**2)) + (2.0 / (sigma_A**2 * K_o**0.28))
        # Dénominateur de B
        den_B = (1.0 / (sigma_A * C_c)**2) - 1.0 - (2.0 / (C_c * sigma_A**2)) + (2.0 / sigma_A**2)
        
        if den_B == 0:
            B = 0.0  # Fallback de sécurité
        else:
            B = num_B / den_B
            
        # --- 5. Multiplicateur final ---
        phi2_con = 1.0 + (rho_l / rho_g - 1.0) * (B * x * (1.0 - x) + x**2)
        return phi2_con
    
    #Get the positive and negative flow areas for a given cell (take into account the two-phase pressure multiplier and friction factor)
    def getAreas(self, i):
        if self.voidFractionTEMP[i] > -0.001:
            # --- FACE POSITIVE (i-1) ---
            rho_m_pos = self.rhoTEMP[i-1]
            rho_l_pos = self.rholTEMP[i-1]
            A_pos = self.areaMatrix[i-1]
            
            # Friction linéaire classique
            loss_fric_pos = (self.getPhi2Phi(i-1)/4) * (self.fTEMP[i-1] / self.D_h[i-1]) * self.DV[i-1]
            
            # Application de la formule : loss = 0.5 * phi2 * K * (rho_m / rho_l) * Area
            #loss_exp_pos = 0.5 * self.getPhi2Expansion(i-1) * self.kexp[i-1] * (rho_m_pos / rho_l_pos) * A_pos
            #loss_con_pos = 0.5 * self.getPhi2Contraction(i-1) * self.kcon[i-1] * (rho_m_pos / rho_l_pos) * A_pos
            
            A_chap_pos = A_pos + loss_fric_pos #+ loss_exp_pos + loss_con_pos

            # --- FACE NÉGATIVE (i) ---
            rho_m_neg = self.rhoTEMP[i]
            rho_l_neg = self.rholTEMP[i]
            A_neg = self.areaMatrix[i]
            
            # Friction linéaire classique
            loss_fric_neg = (self.getPhi2Phi(i)/4) * (self.fTEMP[i] / self.D_h[i]) * self.DV[i]
            
            # Application de la formule
            #loss_exp_neg = 0.5 * self.getPhi2Expansion(i) * self.kexp[i] * (rho_m_neg / rho_l_neg) * A_neg
            #loss_con_neg = 0.5 * self.getPhi2Contraction(i) * self.kcon[i] * (rho_m_neg / rho_l_neg) * A_neg
            
            A_chap_neg = A_neg - loss_fric_neg # - loss_exp_neg - loss_con_neg
            
        return A_chap_pos, A_chap_neg

    #Get the enthalpy values for the liquid and vapor phases at a given pressure
    def getPhasesEnthalpy(self, i):
        P = self.P[i]
        return FAST_IAPWS.get_hl(P*(10**(-6))), FAST_IAPWS.get_hg(P*(10**(-6)))
    
    #Get the Reynolds number for flow in a given cell
    def getReynoldsNumber(self, i):
        U = self.U[i]
        rho = self.rhoTEMP[i]
        P = self.P[i]
        alpha = self.voidFractionTEMP[i]
        ml = FAST_IAPWS.get_mul(P*(10**(-6)))
        mv = FAST_IAPWS.get_mug(P*(10**(-6)))
        m = (mv * ml) / ( ml * (1 - alpha) + mv * alpha )
        
        return rho * abs(U) * self.D_h[i] / m
    
    #Get the Reynolds number for the liquid phase in a given cell
    def getReynoldsNumberLiquid(self, i):
        Ul = self.getUl(i)
        rho = self.rholTEMP[i]
        P = self.P[i]
        m = FAST_IAPWS.get_mul(P*(10**(-6)))
        return rho * abs(Ul) * self.D_h[i] / m
    
    def getReynoldsNumberLiquidD5(self, i):
        Um = self.U[i]
        rhom = self.rhoTEMP[i]
        P = self.P[i]
        xfl = self.getQuality(i)
        m = FAST_IAPWS.get_mul(P*(10**(-6)))
        return rhom * abs(Um) * self.D_h[i] * (1-xfl) / m

    def getReynoldsNumberLiquidOnly(self, i):
        Um = self.U[i]
        rhom = self.rhoTEMP[i]
        P = self.P[i]
        m = FAST_IAPWS.get_mul(P*(10**(-6)))
        return rhom * abs(Um) * self.D_h[i] / m

    #Get the Reynolds number for the vapor phase in a given cell
    def getReynoldsNumberVapor(self, i):
        Ug = self.getUg(i)
        rho = self.rhogTEMP[i]
        P = self.P[i]
        m = FAST_IAPWS.get_mug(P*(10**(-6)))
        return rho * abs(Ug) * self.D_h[i] / m
    
    #Get the liquid velocity in a given cell
    def getUl(self, i):
        if self.voidFractionTEMP[i] >= 0.999:
            return self.U[i]
        return self.U[i] - (self.voidFractionTEMP[i] / ( 1 - self.voidFractionTEMP[i])) * (self.rhogTEMP[i] / self.rhoTEMP[i]) * self.VgjPrimeTEMP[i]
    
    #Get the vapor velocity in a given cell
    def getUg(self, i):
        return self.U[i] + (self.rholTEMP[i] / self.rhoTEMP[i]) * self.VgjPrimeTEMP[i]

    #Get the two-phase pressure multiplier for a given cell
    def lockhartMartinelli(self, i):
        Ul = self.getUl(i)
        Ug = self.getUg(i)
        Um = self.U[i]
        rhom = self.rhoTEMP[i]
        rho_l = self.rholTEMP[i]
        rho_g = self.rhogTEMP[i]
        """
        ## Clement's correlation 
        X = np.sqrt(rho_l/rho_g)*(Ul/Ug)
        return 1 + 1/np.sqrt(X) + 20/X
        ## End Clement's correlation
        """
        ## Modify for the Lockhart-Martinelli correlation present in DONJON5
        # LOCKHART-MARTINELLI CORRELATION
        xth = self.getQuality(i)
        if xth < 1e-5:
            PHIL0 = 1.0
        elif xth > 0.999:
            PHIL0 = (rho_l / rho_g) if rho_g > 0 else 1.0
        else:
            mu_g = FAST_IAPWS.get_mug(self.P[i]*(10**(-6)))
            mu_l = FAST_IAPWS.get_mul(self.P[i]*(10**(-6)))
            XLM = ((1-xth)/xth)**0.9*(rho_g/rho_l)**0.5*(mu_g/mu_l)**0.1
            PHIL0 = 1.0 + 20/XLM + 1.0/XLM**2
        return PHIL0
            
    
        """ mul = IAPWS97(P = self.P[i]*(10**(-6)), x = 0).Liquid.mu
        mulg = IAPWS97(P = self.P[i]*(10**(-6)), x = 1).Vapor.mu

        Rel = self.getReynoldsNumberLiquid(i)
        Reg = self.getReynoldsNumberVapor(i)
        fliq = 0.079 * (Rel)**(-0.25)
        fgas = 0.079 * (Reg)**(-0.25)

        return ((1.2*(rho_l/rho_g - 1)*self.xThTEMP[i]**0.824 + 1)*(rhom/rho_l)*(rho_l/rho_g))**2 #*self.xThTEMP[i] + 1)**0.25
        X = (fliq*rho_l*Ul**2)/(fgas*rho_g*Ug**2)
        return 1+20/X """
    
    def friedel(self, i):
        """
        Multiplicateur diphasique de frottement (Corrélation de Friedel)
        Note : Cette corrélation renvoie phi^2_lo (Liquid-Only).
        """
        x = self.xThTEMP[i]
        epsilon = self.voidFractionTEMP[i]
        rho_l = self.rholTEMP[i]
        rho_g = self.rhogTEMP[i]
        P_MPa = self.P[i] * 1e-6

        # Sécurité numérique pour le monophasique
        if x <= 1e-5 or epsilon <= 1e-5:
            return 1.0

        # Propriétés thermodynamiques
        mu_l = FAST_IAPWS.get_mul(P_MPa)
        mu_g = FAST_IAPWS.get_mug(P_MPa)
        sigma = FAST_IAPWS.get_sigma(P_MPa)

        # Flux massique G (kg/m^2/s) et Diamètre hydraulique
        G = self.rhoTEMP[i] * abs(self.U[i])
        D_h = self.D_h[i]

        # Densité homogène (Équation 41)
        rho_H = 1.0 / (x / rho_g + (1.0 - x) / rho_l)

        # Nombres adimensionnels (Froude et Weber)
        Fr = (G**2) / (self.g * D_h * rho_H**2)
        We = (G**2 * D_h) / (sigma * rho_H)

        # Coefficients de friction de McAdams (Liquid-only et Gas-only)
        Re_lo = G * D_h / mu_l
        Re_go = G * D_h / mu_g
        f_lo = 0.079 * (Re_lo)**(-0.25)
        f_go = 0.079 * (Re_go)**(-0.25)

        # Paramètres intermédiaires E, F, H
        E = (1.0 - x)**2 + x**2 * (rho_l * f_go) / (rho_g * f_lo)
        F = x**0.78 * (1.0 - x)**0.224
        H_param = (rho_l / rho_g)**0.91 * (mu_g / mu_l)**0.19 * (1.0 - mu_g / mu_l)**0.7

        # Multiplicateur final (Équation 40)
        phi2_lo = E + (3.24 * F * H_param) / (Fr**0.045 * We**0.035)
        return phi2_lo
    
    #get the velocity values for a given cell
    def getVelocity(self):
        Ul = []
        Ug = []
        for i in range(self.nCells):
            Ul.append(self.getUl(i))
            Ug.append(self.getUg(i))
        return Ul, Ug
    
    def getDTSUB(self, i):
        """
        From IGE409 equations 33.20 and 3.21. Calculates deltaTsub,D at the OFDB point with the Saha-Zuber correlation.
        """
        P = self.P[i] * 1e-6 
        C_l = FAST_IAPWS.get_cpl(P) * 1000.0
        k_l = FAST_IAPWS.get_kl(P)
        Q = self.rhoTEMP[i] * abs(self.U[i])
        D_h = self.D_h[i]
        idx = min(i, len(self.q__) - 1)  # Ensure index is within bounds
        phi = (self.q__[idx] * self.areaMatrix[idx])/self.phs[idx]
        Pe =(Q*D_h*C_l)/k_l
        if Pe> 70000:
            DT_sub_D = phi/(0.0065*Q*C_l)
        else:
            DT_sub_D = (phi*D_h)/(455.0*k_l)
        return DT_sub_D
