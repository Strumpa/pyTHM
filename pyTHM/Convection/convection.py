# This file contains the implementation of the drift flux model for the THM prototype
# This class models the dynamic and steady-state behavior of a two-phase flow system in different geometries (square and cylindrical channels). 
# It discretizes the channel geometry and sets up the necessary fields for fluid flow, pressure, enthalpy, and void fraction in a thermal-hydraulic model. 
# It also includes methods for transient and steady-state simulations using various numerical techniques. The class supports setting up fission power, initializing 
# the flow fields, creating systems of equations for both steady and transient analysis, and solving them using finite volume method (FVM). Visualization tools 
# are provided to track residuals and convergence during iterative solving.

# Authors : Clement Huet
# Python3 class part of THM_prototype
# uses : - Drift flux model for two-phase flow
#        - Finite volume method for discretization of the conservation equations
#        - IAPWS97 for water properties
#        - THM_linalg for numerical resolution, it include a newton simple iteration method, a Gauss Siedel method, a BiCGStab method and a BiCG method
#        - THM_waterProp for water properties, the calculation of the void fraction, the calculation of the friction factor and the two-phase mltp depend on correlations
#        - THM_plotting for plotting

import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from pyTHM.Solver.linalg import FVM, numericalResolution
from pyTHM.WaterProperties.waterProperties import statesVariables
from pyTHM.Lissage.smooth import if_lisse, max_lisse, min_lisse
from pyTHM.WaterProperties.waterProperties import FastIAPWS
import cProfile

class DFMclass():
    def __init__(self, channel_type, nCells, tInlet, qFlow, pOutlet, height, fuelRadius, cladRadius, pin_pitch, pitch,  numericalMethod, frfaccorel, P2P2corel, voidFractionCorrel, FAST_IAPWS, dt = 0, t_tot = 0, D_h = 0, volumetricArea = 0, porosities=None, acools=None, dhs=None, phs=None, kexp=None, kcon=None, rsin=None):
        
        """
        Attributes:
        - nFaces: Number of discretized cells.
        - pOutlet (Pa), tInlet (K): Inlet velocity, outle t pressure, and inlet enthalpy.
        - qFlow: Mass flow rate of the fluid (kg/s).
        - height (m), fuelRadius (m), cladRadius (m): Geometry of the channel (length, fuel, and clad radii).
        - pin_pitch: Distance between fuel pins in the assembly (m).
        - pitch: Channel width or distance, depending on the geometry.
        - channel_type: Geometry type of the channel, either 'square' or 'cylindrical'.
        - numericalMethod: Chosen method for numerical resolution (e.g., Gauss-Seidel, FVM, BiGStab).
        - voidFractionCorrel, frfaccorel, P2Pcorel: Correlations used for void fraction and other flow properties.
        - FAST_IAPWS : FastIAPWS object with thermo-physical quantities pre-tabulated.
        - dt, t_tot: Time-step and total simulation time for transient analysis.
        - porosities: Array of porosity values along the channel.
        - acools: Array of flow areas for the coolant along the channel (m²).
        - dhs: Array of hydraulic diameters along the channel (m).
        - phs: Array of heating perimeters along the channel (m).
        - kexp, kcon: Arrays of singular pressure drop coefficients for expansion and contraction along the channel.
        - rsin: Array of flow area ratios (small/large) along the channel.
        """

        """
        Methods:
        - __init__(...): Initializes the class with geometric, inlet, outlet, and other user-specified parameters. It also sets up physical constants and mesh properties.
        - set_Fission_Power(Ptot, axial_p_form, Fpow): Sets the fission power source distribution term for the system.
        - get_Fission_Power(): Returns the source term (fission power) distribution along the channel.
        - setInitialFields(): Initializes the primary variables (velocity, pressure, enthalpy, void fraction) for steady-state simulation and updates flow properties.
        - createSystem(): Constructs the system of equations for solving the steady-state two-phase flow problem using the finite volume method.
        - createSystemTransient(): Sets up the system of equations for transient simulation.
        - calculateResiduals(): Calculates the residuals for velocity, pressure, and void fraction, and monitors the convergence.
        - testConvergence(k): Checks if the solution has converged based on residuals after iteration k.
        - residualsVisu(): Updates and visualizes residuals during the iterative solving process.
        - update_jump_source(): Updates the source term for momentum to take into account singular pressure drops due to expansions and contractions in the channel.
        - resolveDFM(): Main function that orchestrates the simulation by calling initializations, solving the system, and managing iterations and convergence criteria.
        - plotResults(): Plots the results of the simulation, including velocity, pressure, enthalpy, and void fraction profiles.
        - setInitialFieldsTransient(): Initializes the fields for transient simulation.
        - compute_T_surf(): Computes the surface temperature on the clad on the enthalpy profile.
        - sousRelaxation(): Implements under-relaxation for the solution update.
        - mergeVAR(): Merges the variables for the system of equations.
        - splitVAR(): Splits the variables for the system of equations.
        - createBoundaryEnthalpy(): Sets the boundary saturation lines for enthalpy.
        """ 

        if __name__ == "__main__":
            cProfile.run('main()', filename='profiling_result.prof')


        #user choice
        self.frfaccorel = frfaccorel
        self.P2Pcorel = P2P2corel
        self.numericalMethod = numericalMethod
        self.voidFractionCorrel = voidFractionCorrel
        self.voidFractionEquation = 'base'

        self.nCells = nCells
        self.nFaces = self.nCells + 1 #Number of faces
        self.pOutlet = pOutlet
        self.tInlet = tInlet

        # Initialize Fast IAPWS tables by pretabulating values around pOutlet
        self.FAST_IAPWS = FAST_IAPWS
        #calculate temporary hInlet
        falsePInlet = pOutlet
        self.hInlet = IAPWS97(T = self.tInlet, P = falsePInlet * 10**(-6)).h*1000 #J/kg

        #Geometry parameters
        self.height = height #m
        self.fuelRadius = fuelRadius #External radius of the fuel m
        self.cladRadius = cladRadius #External radius of the clad m
        self.pitch = pitch
        self.wall_dist = pitch
        self.pin_pitch = pin_pitch
        self.channelType = channel_type

        #Porous media parameters
        self.poro = np.zeros(self.nFaces)
        self.areaMatrix = np.zeros(self.nFaces)
        self.acools = acools
        self.D_h = np.zeros(self.nFaces)
        self.phs = np.zeros(self.nFaces)

        self.poro[0] = porosities[0]
        self.areaMatrix[0] = acools[0]
        self.D_h[0] = dhs[0]
        self.phs[0] = phs[0]
        for i in range(1, self.nCells):
            self.poro[i] = (porosities[i-1]+porosities[i])/2.0
            self.areaMatrix[i] = (acools[i-1]+acools[i])/2.0    
            self.D_h[i] = (dhs[i-1]+dhs[i])/2.0
            self.phs[i] = (phs[i-1]+phs[i])/2.0
        self.poro[self.nCells] = porosities[-1]
        self.areaMatrix[self.nCells] = acools[-1]
        self.D_h[self.nCells] = dhs[-1]
        self.phs[self.nCells] = phs[-1]

        if self.cladRadius > 0.0:
            self.number_of_pins = np.array(phs) / (2*np.pi*self.cladRadius)
        else: 
            self.number_of_pins = 0.0

        #compute temporary uInlet
        self.qFlow = qFlow #kg/s
        self.rhoInlet = IAPWS97(T = self.tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
        self.uInlet = self.qFlow / (self.areaMatrix[0] * self.rhoInlet) #m/s

        self.DV = (self.height/self.nCells) * self.areaMatrix #Volume of the control volume m3


        self.Dz = self.height/self.nCells #Height of the control volume m
        self.z_mesh = np.linspace(0, self.height, self.nFaces)
        self.epsilonTarget = 0.1
        self.K_loss = 0.0


        self.kexp_face = np.zeros(self.nFaces)
        self.kcon_face = np.zeros(self.nFaces)
        self.rsin_face = np.ones(self.nFaces)
        if kexp is not None and kcon is not None and rsin is not None:
            self.kexp_face[0] = kexp[0]
            self.kcon_face[0] = kcon[0]
            self.rsin_face[0] = rsin[0]
            for i in range(1, self.nCells):
                self.kexp_face[i] = (kexp[i-1] + kexp[i]) / 2.0
                self.kcon_face[i] = (kcon[i-1] + kcon[i]) / 2.0
                self.rsin_face[i] = (rsin[i-1] + rsin[i]) / 2.0
            self.kexp_face[-1] = kexp[-1]
            self.kcon_face[-1] = kcon[-1]
            self.rsin_face[-1] = rsin[-1]
        
        self.S_mass = np.zeros(self.nFaces)
        self.S_mom = np.zeros(self.nFaces)
        self.S_h = np.zeros(self.nFaces)

        self.epsInnerIteration = 1e-4
        self.maxInnerIteration = 1000
        if self.numericalMethod == 'BiCGStab' or self.numericalMethod == 'BiCG' or self.numericalMethod == 'FVM':
            self.sousRelaxFactor = 0.8
        else:
            self.sousRelaxFactor = 1
        self.epsOuterIteration = 1e-4
        self.maxOuterIteration = 50

        #Universal constant
        self.g = 9.81 #m/s^2
        self.R = 8.314 #J/(mol*K)

        #residuals
        self.EPSresiduals = []
        self.rhoResiduals = []
        self.rhoGResiduals = []
        self.rhoLResiduals = []
        self.xThResiduals = []
        self.UResiduals = []
        self.Iteration = []
        self.I = []

        self.hlSat = []
        self.hgSat = []

        #Transient parameters
        self.dt = dt
        if dt != 0:
            self.t_tot = t_tot
            self.timeList = np.arange(0, self.t_tot, self.dt)
            self.timeCount = 0
    
    #Fission power
    def set_Fission_Power(self, Ptot, axial_p_forms, Fpow):

        dv = self.pitch**2*(self.height / self.nCells) #m

        self.q__ = np.zeros(self.nCells)
        self.QFUEL = np.zeros(self.nCells)

        if Ptot > 0.0:
            Power_dist = (Ptot) * axial_p_forms / self.nCells #W Axial power distribution
            linear_powers = Power_dist
            assembly_section = self.pitch**2
            proportion_fuel = assembly_section / (self.number_of_pins*(np.pi * self.fuelRadius**2)) #m2 / m2
            fraction_in_fuel = Fpow*proportion_fuel #Fraction of the total power released in fuel
            fraction_in_coolant = (1.0-Fpow)*proportion_fuel # Fraction of the total power released in coolant
            if self.dt == 0:
                self.QFUEL = Power_dist*fraction_in_fuel / dv #W/m3
                phi2 = 0.5*self.QFUEL*self.fuelRadius**2 / self.cladRadius
                self.q__ = np.zeros(self.nCells)
                for i in range(self.nCells):
                    self.q__[i] = phi2[i]*self.phs[i] / self.areaMatrix[i]
            if self.dt != 0: # This option is not suppoted yet, dummy initialization for now
                t_final_q = 0
                self.q__ = np.zeros((len(self.timeList), len(axial_p_forms)))
                for t, time in enumerate(self.timeList):
                    if time < t_final_q:
                        self.q__[t] = (self.timeList[t]/t_final_q)*(phi2[i]*self.phs[i]/self.areaMatrix[i]) #W/m3
                    else:
                        self.q__[t] = Power_dist*fraction_in_fuel / dv #W/m3
            

    #Recovers the fission power distribution
    def get_QFUEL(self):
        """
        function to retrieve the fission power density distribution in the fuel rod
        """
        return self.QFUEL
    
    def update_sources(self, S_mass, S_mom, S_h):
        """Updates the lateral source terms (cross-flow and conduction) before each iteration"""
        self.S_mass = S_mass
        self.S_mom = S_mom
        self.S_h = S_h

    def get_Fission_Power(self):
        """
        function to retrieve a given source term from the axial profile used to model fission power distribution in the fuel rod
        """
        return self.q__
        
    #Define the initial fields for the first iteration
    def setInitialFields(self):
        
        #Steady state
        if self.dt == 0:
        
            self.U = [np.ones(self.nFaces)*self.uInlet] #
            self.P = [np.ones(self.nFaces)*self.pOutlet] #
            self.H = [np.ones(self.nFaces)*self.hInlet] #
            self.voidFraction = [np.array([i*self.epsilonTarget/self.nFaces for i in range(self.nFaces)])]

            updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.cladRadius, self.pin_pitch, self.pitch, self.D_h, self.areaMatrix, self.poro, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz, self.q__, self.phs, self.qFlow, self.fuelRadius, self.pitch/2, self.rsin_face, self.FAST_IAPWS)
            updateVariables.createFields()
                
            self.xTh = [np.ones(self.nFaces)]
            self.rhoL= [updateVariables.rholTEMP]
            self.rhoG = [updateVariables.rhogTEMP]
            self.rho = [updateVariables.rhoTEMP]
            self.Dhfg = [updateVariables.DhfgTEMP]
            self.f = [updateVariables.fTEMP]
            self.areaMatrix_1 = [updateVariables.areaMatrix_1TEMP]
            self.areaMatrix_2 = [updateVariables.areaMatrix_2TEMP]
            #self.areaMatrix = updateVariables.areaMatrixTEMP
            self.Vgj = [updateVariables.VgjTEMP]
            self.C0 =[updateVariables.C0TEMP]
            self.VgjPrime = [updateVariables.VgjPrimeTEMP]

        #Transient
        else:
            if self.timeCount == 0:
                self.U = [self.velocityList[self.timeCount]]
                self.P = [self.pressureList[self.timeCount]]
                self.H = [self.enthalpyList[self.timeCount]]
                self.voidFraction = [self.voidFractionList[self.timeCount]]

                updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.cladRadius, self.pin_pitch, self.pitch, self.D_h, self.areaMatrix, self.poro, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz, self.q__, self.phs, self.qFlow, self.fuelRadius, self.pitch/2, self.rsin_face, self.FAST_IAPWS)
                updateVariables.createFields()

                self.xTh = [np.ones(self.nFaces)]
                self.rhoL= [updateVariables.rholTEMP]
                self.rhoG = [updateVariables.rhogTEMP]
                self.rho = [updateVariables.rhoTEMP]
                self.Dhfg = [updateVariables.DhfgTEMP]
                self.f = [updateVariables.fTEMP]
                self.areaMatrix_1 = [updateVariables.areaMatrix_1TEMP]
                self.areaMatrix_2 = [updateVariables.areaMatrix_2TEMP]
                #self.areaMatrix = updateVariables.areaMatrixTEMP
                self.Vgj = [updateVariables.VgjTEMP]
                self.C0 =[updateVariables.C0TEMP]
                self.VgjPrime = [updateVariables.VgjPrimeTEMP]

                self.xThList[self.timeCount] = self.xTh[-1]
                self.rhoList[self.timeCount] = self.rho[-1]
                self.rhoGList[self.timeCount] = self.rhoG[-1]
                self.rhoLList[self.timeCount] = self.rhoL[-1]
                self.DhfgList[self.timeCount] = self.Dhfg[-1]
                self.fList[self.timeCount] = self.f[-1]
                self.areaMatrix_1List[self.timeCount] = self.areaMatrix_1[-1]
                self.areaMatrix_2List[self.timeCount] = self.areaMatrix_2[-1]
                self.areaMatrixList[self.timeCount] = self.areaMatrix
                self.VgjList[self.timeCount] = self.Vgj[-1]
                self.C0List[self.timeCount] = self.C0[-1]
                self.VgjPrimeList[self.timeCount] = self.VgjPrime[-1]
            
            if self.timeCount != 0:
                self.U = [self.velocityList[self.timeCount-1]]
                self.P = [self.pressureList[self.timeCount-1]]
                self.H = [self.enthalpyList[self.timeCount-1]]
                self.voidFraction = [self.voidFractionList[self.timeCount-1]]
                self.xTh = [self.xThList[self.timeCount-1]]
                self.rhoL = [self.rhoLList[self.timeCount-1]]
                self.rhoG = [self.rhoGList[self.timeCount-1]]
                self.rho = [self.rhoList[self.timeCount-1]]
                self.Dhfg = [self.DhfgList[self.timeCount-1]]
                self.f = [self.fList[self.timeCount-1]]
                self.areaMatrix_1 = [self.areaMatrix_1List[self.timeCount-1]]
                self.areaMatrix_2 = [self.areaMatrix_2List[self.timeCount-1]]
                self.areaMatrix = self.areaMatrixList[self.timeCount-1]
                self.Vgj = [self.VgjList[self.timeCount-1]]
                self.C0 = [self.C0List[self.timeCount-1]]
                self.VgjPrime = [self.VgjPrimeList[self.timeCount-1]]

    #Create the matrix for the velocity and pressure coupling resolution equation system
    def createSystemVelocityPressure(self):

        U_old = self.U[-1]
        P_old = self.P[-1]
        H_old = self.H[-1]
        epsilon_old = self.voidFraction[-1]
        rho_old = self.rho[-1]
        rho_g_old = self.rhoG[-1]
        rho_l_old = self.rhoL[-1]
        areaMatrix = self.areaMatrix
        areaMatrix_old_1 = self.areaMatrix_1[-1]
        areaMatrix_old_2 = self.areaMatrix_2[-1]
        Dhfg = self.Dhfg[-1]
        x_th_old = self.xTh[-1]
        f = self.f[-1]
        V_gj_old = self.Vgj[-1]
        Vgj_prime = self.VgjPrime[-1]
        C0 = self.C0[-1]

        VAR_old = self.mergeVar(U_old,P_old)
        rho_old = self.mergeVar(rho_old, rho_old)
        rho_g_old = self.mergeVar(rho_g_old, rho_g_old)
        rho_l_old = self.mergeVar(rho_l_old, rho_l_old)
        epsilon_old = self.mergeVar(epsilon_old, epsilon_old)
        areaMatrix = self.mergeVar(areaMatrix, areaMatrix)
        areaMatrix_old_1 = self.mergeVar(areaMatrix_old_1, areaMatrix_old_1)
        areaMatrix_old_2 = self.mergeVar(areaMatrix_old_2, areaMatrix_old_2)
        V_gj_old = self.mergeVar(V_gj_old, V_gj_old)
        Vgj_prime = self.mergeVar(Vgj_prime, Vgj_prime)
        Dhfg = self.mergeVar(Dhfg, Dhfg)
        C0 = self.mergeVar(C0, C0)
        x_th_old = self.mergeVar(x_th_old, x_th_old)
        
        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = 0, Am1 = 1, D0 = self.uInlet, Dm1 = self.pOutlet, N_vol = 2*self.nFaces, H = self.height)
        VAR_VFM_Class.boundaryFilling()

        for i in range(1, 2*self.nFaces-1):
            #Inside the velocity submatrix
            if i < self.nFaces-1:
                VAR_VFM_Class.set_ADi(i, ci = - rho_old[i-1]*areaMatrix[i-1],
                ai = rho_old[i]*areaMatrix[i],
                bi = 0,
                di =  self.S_mass[i])
            elif i == self.nFaces-1:
                VAR_VFM_Class.set_ADi(i, 
                ci = - rho_old[i-1]*areaMatrix[i-1],
                ai = rho_old[i]*areaMatrix[i],
                bi = 0,
                di =  self.S_mass[i])

            #Inside the pressure submatrix
            elif i >= self.nFaces and i < 2*self.nFaces-1:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(i, ci = 0,
                ai = - areaMatrix[i],
                bi = areaMatrix[i],
                di = - (((rho_old[i+1]+ rho_old[i])* self.g/2) * self.DV[i%self.nFaces] / 2) + DI + self.S_mom[i%self.nFaces])
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nFaces,
                ai = - rho_old[i]*VAR_old[i-self.nFaces]*areaMatrix_old_2[i],
                bi = rho_old[i+1]*VAR_old[i+1-self.nFaces]*areaMatrix_old_1[i+1])        
        self.FVM = VAR_VFM_Class

    #Create the enthalpy matrix resolution equation system
    def createSystemEnthalpy(self):

        U_old = self.U[-1]
        P_old = self.P[-1]
        H_old = self.H[-1]
        epsilon_old = self.voidFraction[-1]
        rho_old = self.rho[-1]
        rho_g_old = self.rhoG[-1]
        rho_l_old = self.rhoL[-1]
        areaMatrix = self.areaMatrix
        Dhfg = self.Dhfg[-1]
        x_th_old = self.xTh[-1]
        f = self.f[-1]
        V_gj_old = self.Vgj[-1]
        Vgj_prime = self.VgjPrime[-1]
        C0 = self.C0[-1]


        i = -1
        DI = (1/2) * (P_old[i-1]*areaMatrix[i-1] - P_old[i]*areaMatrix[i]) * ((U_old[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (U_old[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
        DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
        DM1 = self.q__[i-1] * self.DV[i] + DI + DI2 +self.S_h[-1]
        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = - rho_old[-2] * U_old[-2] * areaMatrix[-2], Am1 = rho_old[-1] * U_old[-1] * areaMatrix[-1], D0 = self.hInlet, Dm1 = DM1, N_vol = self.nFaces, H = self.height)
        VAR_VFM_Class.boundaryFilling()
        for i in range(1,self.nFaces -1):
            #Inside the enthalpy submatrix
            DI = (1/2) * (P_old[i-1]*areaMatrix[i-1] - P_old[i]*areaMatrix[i]) * ((U_old[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (U_old[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
            DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
            VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * U_old[i-1] * areaMatrix[i-1],
                ai = rho_old[i] * U_old[i] * areaMatrix[i],
                bi = 0,
                di =  self.q__[i-1] * self.DV[i] + DI2 + DI + self.S_h[i])
        
        self.FVM = VAR_VFM_Class

    #Create the enthalpy matrix for the transient resolution equation system
    def createSystemEnthalpyTransient(self):

        U_old = self.U[-1]
        P_old = self.P[-1]
        H_old = self.H[-1]
        epsilon_old = self.voidFraction[-1]
        rho_old = self.rho[-1]
        rho_g_old = self.rhoG[-1]
        rho_l_old = self.rhoL[-1]
        areaMatrix = self.areaMatrix
        Dhfg = self.Dhfg[-1]
        x_th_old = self.xTh[-1]
        f = self.f[-1]
        V_gj_old = self.Vgj[-1]
        Vgj_prime = self.VgjPrime[-1]
        C0 = self.C0[-1]


        i = -1
        DI = (1/2) * (P_old[i]*areaMatrix[i] - P_old[i-1]*areaMatrix[i-1]) * ((U_old[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (U_old[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
        DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
        DT1 = - (self.pressureList[self.timeCount][i%self.nFaces] * self.areaMatrix[i] - P_old[i] * areaMatrix[i])*(self.Dz/self.dt) + (self.rhoList[self.timeCount][i%self.nFaces] * self.enthalpyList[self.timeCount][i%self.nFaces] * areaMatrix[i] * (self.Dz / self.dt))
        DM1 = self.q__[self.timeCount][i-1] * self.DV[i] + DI + DI2 + DT1 +self.S_h[-1]
        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = - rho_old[-2] * U_old[-2] * areaMatrix[-2] + rho_old[-2] * areaMatrix[-2] * (self.Dz / self.dt), Am1 = rho_old[-1] * U_old[-1] * areaMatrix[-1], D0 = self.hInlet, Dm1 = DM1, N_vol = self.nFaces, H = self.height)
        VAR_VFM_Class.boundaryFilling()
        for i in range(1,self.nFaces -1):
            #Inside the enthalpy submatrix
            DI = (1/2) * (P_old[i]*areaMatrix[i] - P_old[i-1]*areaMatrix[i-1]) * ((U_old[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (U_old[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
            DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
            DT1 = -(self.pressureList[self.timeCount][i%self.nFaces] * self.areaMatrixList[self.timeCount][i%self.nFaces] - P_old[i] * areaMatrix[i])*(self.Dz/self.dt) + (self.rhoList[self.timeCount][i%self.nFaces] * self.enthalpyList[self.timeCount][i%self.nFaces] * areaMatrix[i] * (self.Dz / self.dt))
            VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * U_old[i-1] * areaMatrix[i-1] + rho_old[i] * areaMatrix[i] * (self.Dz / self.dt),
                ai = rho_old[i] * U_old[i] * areaMatrix[i],
                bi = 0,
                di =  self.q__[self.timeCount][i-1] * self.DV[i] + DI + DI2 + DT1 + self.S_h[i])
        
        self.FVM = VAR_VFM_Class

    #Create the transient matrix for the velocity and pressure coupling resolution equation system
    def createSystemVelocityPressureTransient(self):

        U_old = self.U[-1]
        P_old = self.P[-1]
        H_old = self.H[-1]
        epsilon_old = self.voidFraction[-1]
        rho_old = self.rho[-1]
        rho_g_old = self.rhoG[-1]
        rho_l_old = self.rhoL[-1]
        areaMatrix = self.areaMatrix
        areaMatrix_old_1 = self.areaMatrix_1[-1]
        areaMatrix_old_2 = self.areaMatrix_2[-1]
        Dhfg = self.Dhfg[-1]
        x_th_old = self.xTh[-1]
        f = self.f[-1]
        V_gj_old = self.Vgj[-1]
        Vgj_prime = self.VgjPrime[-1]
        C0 = self.C0[-1]

        VAR_old = self.mergeVar(U_old,P_old)
        rho_old = self.mergeVar(rho_old, rho_old)
        rho_g_old = self.mergeVar(rho_g_old, rho_g_old)
        rho_l_old = self.mergeVar(rho_l_old, rho_l_old)
        epsilon_old = self.mergeVar(epsilon_old, epsilon_old)
        areaMatrix = self.mergeVar(areaMatrix, areaMatrix)
        areaMatrix_old_1 = self.mergeVar(areaMatrix_old_1, areaMatrix_old_1)
        areaMatrix_old_2 = self.mergeVar(areaMatrix_old_2, areaMatrix_old_2)
        V_gj_old = self.mergeVar(V_gj_old, V_gj_old)
        Vgj_prime = self.mergeVar(Vgj_prime, Vgj_prime)
        Dhfg = self.mergeVar(Dhfg, Dhfg)
        C0 = self.mergeVar(C0, C0)
        x_th_old = self.mergeVar(x_th_old, x_th_old)

        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = 0, Am1 = 1, D0 = self.uInlet, Dm1 = self.pOutlet, N_vol = 2*self.nFaces, H = self.height)
        VAR_VFM_Class.boundaryFilling()
        for i in range(1, 2*self.nFaces-1):
            #Inside the velocity submatrix
            if i < self.nFaces-1:
                VAR_VFM_Class.set_ADi(i, ci = - rho_old[i-1]*areaMatrix[i-1],
                ai = rho_old[i]*areaMatrix[i],
                bi = 0,
                di =  ( self.rhoList[self.timeCount][i%self.nFaces] *areaMatrix[i] - rho_old[i] *areaMatrix[i] ) * (self.Dz / self.dt) + self.S_mass[i])
            elif i == self.nFaces-1:
                VAR_VFM_Class.set_ADi(i, 
                ci = - rho_old[i-1]*areaMatrix[i-1],
                ai = rho_old[i]*areaMatrix[i],
                bi = 0,
                di =  ( self.rhoList[self.timeCount][i%self.nFaces] * areaMatrix[i] - rho_old[i] *areaMatrix[i] ) * (self.Dz / self.dt) + self.S_mass[i])

            #Inside the pressure submatrix
            elif i == self.nFaces:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(self.nFaces, 
                ci = 0,
                ai = - areaMatrix[i],
                bi = areaMatrix[i],
                di = - ((rho_old[i+1]+ rho_old[i])* self.g * self.DV[i%self.nFaces] / 2) + DI + (self.rhoList[self.timeCount][i%self.nFaces] * areaMatrix[i] * self.velocityList[self.timeCount][i%self.nFaces] * (self.Dz / self.dt))+self.S_mom[i%self.nFaces])
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nFaces,
                ai = - rho_old[i]*VAR_old[i-self.nFaces]*areaMatrix_old_2[i] + rho_old[i]*areaMatrix[i]*(self.Dz/self.dt),
                bi = rho_old[i+1]*VAR_old[i-self.nFaces+1]*areaMatrix_old_1[i+1])

            elif i > self.nFaces and i < 2*self.nFaces-1:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(i, ci = 0,
                ai = - areaMatrix[i],
                bi = areaMatrix[i],
                di = - ((rho_old[i+1]+ rho_old[i])* self.g * self.DV[i%self.nFaces] / 2) + DI + (self.rhoList[self.timeCount][i%self.nFaces] * areaMatrix[i] * self.velocityList[self.timeCount][i%self.nFaces] * (self.Dz / self.dt))+self.S_mom[i%self.nFaces])
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nFaces,
                ai = - rho_old[i]*VAR_old[i-self.nFaces]*areaMatrix_old_2[i] + rho_old[i]*areaMatrix[i]*(self.Dz/self.dt),
                bi = rho_old[i+1]*VAR_old[i+1-self.nFaces]*areaMatrix_old_1[i+1])

        self.FVM = VAR_VFM_Class

    #Calculate the residuals
    def calculateResiduals(self):# updates the residuals
        self.EPSresiduals.append(np.linalg.norm(self.voidFraction[-1] - self.voidFraction[-2]))
        self.rhoResiduals.append(np.linalg.norm((self.rho[-1] - self.rho[-2])/self.rho[-1]))
        self.xThResiduals.append(np.linalg.norm(self.xTh[-1] - self.xTh[-2]))

    #Checking for convergence
    def testConvergence(self, k):# does not change anything and returns a boolean
        print(f'Convergence test number {k}, RES: errEPS: {self.EPSresiduals[-1]}, errRHO: {self.rhoResiduals[-1]}, errQua: {self.xThResiduals[-1]}')
        if self.EPSresiduals[-1] < self.epsOuterIteration and self.rhoResiduals[-1] < self.epsOuterIteration: #and self.xThResiduals[-1] < 1e-3 :
            #print(f'Convergence test number {k}, RES: errEPS: {self.EPSresiduals[-1]}, errRHO: {self.rhoResiduals[-1]}, errQua: {self.xThResiduals[-1]}')
            return True
        else:
            return False

    #Creting the initial fields for the transient resolution
    def setInitialFieldsTransient(self):
        
        self.velocityList = np.zeros((len(self.timeList), self.nFaces))
        self.pressureList = np.zeros((len(self.timeList), self.nFaces))
        self.enthalpyList = np.zeros((len(self.timeList), self.nFaces))
        self.voidFractionList = np.zeros((len(self.timeList), self.nFaces))
        self.rhoList = np.zeros((len(self.timeList), self.nFaces))
        self.rhoGList = np.zeros((len(self.timeList), self.nFaces))
        self.rhoLList = np.zeros((len(self.timeList), self.nFaces))
        self.xThList = np.zeros((len(self.timeList), self.nFaces))
        self.DhfgList = np.zeros((len(self.timeList), self.nFaces))
        self.fList = np.zeros((len(self.timeList), self.nFaces))
        self.areaMatrix_1List = np.zeros((len(self.timeList), self.nFaces))
        self.areaMatrix_2List = np.zeros((len(self.timeList), self.nFaces))
        self.areaMatrixList = np.zeros((len(self.timeList), self.nFaces))
        self.VgjList = np.zeros((len(self.timeList), self.nFaces))
        self.C0List = np.zeros((len(self.timeList), self.nFaces))
        self.VgjPrimeList = np.zeros((len(self.timeList), self.nFaces))

        self.velocityList[:,0] = self.uInlet
        self.pressureList[:,-1] = self.pOutlet
        self.enthalpyList[:,0] = self.hInlet
        self.velocityList[0,:] = self.uInlet
        self.pressureList[0,:] = self.pOutlet
        self.enthalpyList[0,:] = self.hInlet

    #Function to visualise the live evolution of the resuaduels
    def residualsVisu(self):
        # Update the line data
        self.line.set_xdata(self.I)
        self.line.set_ydata(self.rhoResiduals)

        # Adjust axis limits if necessary
        self.ax.relim()         # Recomputes data limits
        self.ax.autoscale_view()  # Automatically readjusts the view

        # Draw the modifications
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    #The pressure inlet change so the inlet velocity and enthalpy need to be updated between the pressure-velocity and the enthalpy resolution
    def updateInlet(self):
        #Update uInlet
        print(f"P[-1][0] = {self.P[-1][0]} before calling IAPWS97", flush=True)
        self.rhoInlet = IAPWS97(T = self.tInlet, P = self.P[-1][0]*10**(-6)).rho #kg/m3
        self.uInlet = self.qFlow / (self.areaMatrix[0] * self.rhoInlet) #m/s
        #Update hInlet
        self.hInlet = IAPWS97(T = self.tInlet, P = self.P[-1][0]*10**(-6)).h*1000 #J/kg

    def update_jump_source(self):
        """
        Updates the source term for the momentum equation to account for numerical artifacts and singular pressure losses.
        """
        self.S_mom = np.zeros(self.nFaces)
        
        # 1. Instantiation of the object for two-phase correlations
        water = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.cladRadius, self.pin_pitch, self.pitch, self.D_h, self.areaMatrix, self.poro, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz, self.q__, self.phs, self.qFlow, self.fuelRadius, self.pitch/2, self.rsin_face, self.FAST_IAPWS)
        # Real-time connection so that the correlations use the correct values
        water.xThTEMP = self.xTh[-1]
        water.voidFractionTEMP = self.voidFraction[-1]
        water.rholTEMP = self.rhoL[-1]
        water.rhogTEMP = self.rhoG[-1]
        water.rhoTEMP = self.rho[-1]
        
        for i in range(1, self.nFaces - 1):
            A1 = self.acools[i-1]
            # The downstream cell is i (except at the last face)
            A2 = self.acools[i] if i < self.nCells else self.acools[-1]
            A_face = self.areaMatrix[i]
            
            u_m = self.U[-1][i]
            rho_m = self.rho[-1][i]
            rho_l = self.rhoL[-1][i] 
            
            # Local mass flow rate through the face
            m_dot = rho_m * u_m * A_face
            
            force_artifact_N = 0.0
            force_perte_N = 0.0
            
            # --- 1. NUMERICAL BERNOULLI ARTIFACT CORRECTION (FVM) ---
            if abs(A1 - A2) > 1e-6:
                # A. What the FVM matrix actually computes by conservative telescoping
                # This is exactly (rho * U_downstream^2 - rho * U_upstream^2)
                dP_fvm = (m_dot**2 / rho_m) * (1.0/(A2**2) - 1.0/(A1**2))
                
                # B. What physics dictates (pure Bernoulli recovery)
                dP_ana = (m_dot**2 / (2.0 * rho_m)) * (1.0/(A2**2) - 1.0/(A1**2))
                
                # The artifact is the native error of the FVM matrix.
                artifact = dP_fvm - dP_ana
                force_artifact_N = artifact * A_face

            # --- 2. EXACT CALCULATION OF SINGULAR PRESSURE LOSSES (K) ---
            if self.kexp_face[i] > 1e-6 or self.kcon_face[i] > 1e-6:
                # The K coefficient from PARCS (or StarterDD) is always calibrated on the downstream cell velocity (A2)
                G_true = m_dot / A2
                facteur_cinetique = 0.5 * (G_true**2) / rho_l
                
                phi2_exp = water.getPhi2Expansion(i)
                phi2_con = water.getPhi2Contraction(i)
                # phi2_exp = 1.0
                # phi2_con = 1.0
                
                dP_perte = (self.kexp_face[i] * phi2_exp + self.kcon_face[i] * phi2_con) * facteur_cinetique
                force_perte_N = dP_perte * A_face
                
            # --- 3. INJECTION INTO THE SOURCE TERM ---
            if abs(force_artifact_N) > 0.0 or abs(force_perte_N) > 0.0:
                # Add the artifact correction and subtract the fluid resistance force
                self.S_mom[i] += force_artifact_N - force_perte_N

    #Main function to solve the drift flux model
    def resolveDFM(self):


        #Steady state
        if self.dt == 0:

            self.setInitialFields()
            # Activate interactive mode
            #plt.ion()
            # Create the figure and axis
            #self.fig, self.ax = plt.subplots()
            # Initialization of the line that will be updated
            #self.line, = self.ax.plot(self.I, self.rhoResiduals, 'r-', marker='o')  # 'r-' for a red line with markers

            #Loop for the outer iterations (velocity-pressure and enthalpy)
            for k in range(self.maxOuterIteration):
                self.update_jump_source()
                self.createSystemVelocityPressure()
                resolveSystem = numericalResolution(self.FVM,self.mergeVar(self.U[-1], self.P[-1]), self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                Utemp, Ptemp = self.splitVar(resolveSystem.x)
                
                self.U.append(Utemp)
                self.P.append(Ptemp)
                
                self.updateInlet()
                
                self.createSystemEnthalpy()
                resolveSystem = numericalResolution(self.FVM, self.H[-1], self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                
                Htemp = resolveSystem.x

                self.H.append(Htemp)
                updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.cladRadius, self.pin_pitch, self.pitch, self.D_h, self.areaMatrix, self.poro, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz, self.q__, self.phs, self.qFlow, self.fuelRadius, self.pitch/2, self.rsin_face, self.FAST_IAPWS)
                updateVariables.updateFields()

                self.xTh.append(updateVariables.xThTEMP)
                self.rhoL.append(updateVariables.rholTEMP)
                self.rhoG.append(updateVariables.rhogTEMP)
                self.rho.append(updateVariables.rhoTEMP)
                self.voidFraction.append(updateVariables.voidFractionTEMP)
                self.Dhfg.append(updateVariables.DhfgTEMP)
                self.f.append(updateVariables.fTEMP)
                self.areaMatrix_1.append(updateVariables.areaMatrix_1TEMP)
                self.areaMatrix_2.append(updateVariables.areaMatrix_2TEMP)
                self.Vgj.append(updateVariables.VgjTEMP)
                self.C0.append(updateVariables.C0TEMP)
                self.VgjPrime.append(updateVariables.VgjPrimeTEMP)

                self.sousRelaxation()
                self.calculateResiduals()
                self.I.append(k)
                #self.residualsVisu()

                convergence = self.testConvergence(k)

                self.updateInlet()

                if convergence == True:
                    print(f'Convergence reached at iteration number: {k}')
                    break

                elif k == self.maxOuterIteration - 1:
                    raise ValueError('Convergence not reached in the resolution of the drift flux model, not enough iterations. k = ', k)


            self.Ul = updateVariables.Ul
            self.Ug = updateVariables.Ug
            self.Rel = updateVariables.Rel
            #plt.ioff()
            #plt.show()
        
        #Transient
        elif self.dt != 0:

            self.setInitialFieldsTransient()
            # Activate interactive mode
            plt.ion()
            # Create the figure and axis
            self.fig, self.ax = plt.subplots()
            self.figh, self.axh = plt.subplots()   
            # Initialization of the line that will be updated
            self.line, = self.ax.plot(self.I, self.rhoResiduals, 'r-', marker='o')  # 'r-' for a red line with markers
            
            for t in range(0, len(self.timeList)-1):
                self.timeCount = t
                self.setInitialFields()

                for k in range(self.maxOuterIteration):
                    self.createSystemVelocityPressureTransient()
                    resolveSystem = numericalResolution(self.FVM,self.mergeVar(self.U[-1], self.P[-1]), self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                    Utemp, Ptemp = self.splitVar(resolveSystem.x)
                    
                    self.U.append(Utemp)
                    self.P.append(Ptemp)
                    self.updateInlet()
                    
                    self.createSystemEnthalpyTransient()
                    resolveSystem = numericalResolution(self.FVM, self.H[-1], self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                    
                    Htemp = resolveSystem.x

                    self.H.append(Htemp)
                    updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.cladRadius, self.pin_pitch, self.pitch, self.D_h, self.areaMatrix, self.poro, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz, self.q__, self.phs, self.qFlow, self.fuelRadius, self.pitch/2, self.rsin_face, self.FAST_IAPWS)
                    updateVariables.updateFields()

                    self.xTh.append(updateVariables.xThTEMP)
                    self.rhoL.append(updateVariables.rholTEMP)
                    self.rhoG.append(updateVariables.rhogTEMP)
                    self.rho.append(updateVariables.rhoTEMP)
                    self.voidFraction.append(updateVariables.voidFractionTEMP)
                    self.Dhfg.append(updateVariables.DhfgTEMP)
                    self.f.append(updateVariables.fTEMP)
                    self.areaMatrix_1.append(updateVariables.areaMatrix_1TEMP)
                    self.areaMatrix_2.append(updateVariables.areaMatrix_2TEMP)
                    self.Vgj.append(updateVariables.VgjTEMP)
                    self.C0.append(updateVariables.C0TEMP)
                    self.VgjPrime.append(updateVariables.VgjPrimeTEMP)
                    #(f'rho: {self.rho[-1]}, U: {self.U[-1]}, P: {self.P[-1]}, H: {self.H[-1]}')
                    self.sousRelaxation()
                    self.calculateResiduals()
                    if self.I == []:
                        self.I.append(0)
                    else:
                        self.I.append(1+self.I[-1])
                    self.residualsVisu()

                    convergence = self.testConvergence(k)

                    self.updateInlet()

                    if convergence == True:
                        self.velocityList[self.timeCount+1] = self.U[-1]
                        self.pressureList[self.timeCount+1] = self.P[-1]
                        self.enthalpyList[self.timeCount+1] = self.H[-1]
                        self.voidFractionList[self.timeCount+1] = self.voidFraction[-1]
                        self.xThList[self.timeCount+1] = self.xTh[-1]
                        self.rhoList[self.timeCount+1] = self.rho[-1]
                        self.rhoGList[self.timeCount+1] = self.rhoG[-1]
                        self.rhoLList[self.timeCount+1] = self.rhoL[-1]
                        self.DhfgList[self.timeCount+1] = self.Dhfg[-1]
                        self.fList[self.timeCount+1] = self.f[-1]
                        self.areaMatrix_1List[self.timeCount+1] = self.areaMatrix_1[-1]
                        self.areaMatrix_2List[self.timeCount+1] = self.areaMatrix_2[-1]
                        self.areaMatrixList[self.timeCount+1] = self.areaMatrix
                        self.VgjList[self.timeCount+1] = self.Vgj[-1]
                        self.C0List[self.timeCount+1] = self.C0[-1]
                        self.VgjPrimeList[self.timeCount+1] = self.VgjPrime[-1]

                        print(f'Convergence reached at iteration number: {k} for time step: {t}')
                        break

                    elif k == self.maxOuterIteration - 1:
                        raise ValueError('Convergence not reached')
                    
                """ champs = [Utemp, Ptemp, Htemp, self.voidFraction[-1], self.rho[-1], self.rhoG[-1]]
                titres = ['U', 'P', 'H', 'epsilon', 'rho', 'rho_g']
                fig, axs = plt.subplots(3, 2, figsize=(12, 8))
                for i, ax in enumerate(axs.flat):
                    ax.plot(self.z_mesh, champs[i])
                    ax.set_title(titres[i])
                    ax.set_xlabel('x')
                plt.tight_layout()
                plt.title(f'Champs pour le pas de temps {t}')
                plt.legend()
                plt.show() """

            plt.ioff()
            plt.show()

            #need to interpolate to get the correct number of values (convert from node-centered to cell-centered)
        
        self.T_water = np.zeros(self.nFaces)
        for i in range(self.nFaces):
            self.T_water[i] = self.FAST_IAPWS.get_sub_T(self.P[-1][i]*10**-6, self.H[-1][i]*10**-3)

        self.interpolate()

    #Function to interpolate the values from the nodes to the cells
    def interpolate(self):
        Ptemp, Utemp, Htemp, voidFractionTemp, rhoTemp, rhoGTemp, rhoLTemp, xThTemp, DhfgTemp, fTemp, areaMatrix_1Temp, areaMatrix_2Temp, areaMatrixTemp, VgjTemp, C0Temp, VgjPrimeTemp = self.P[-1], self.U[-1], self.H[-1], self.voidFraction[-1], self.rho[-1], self.rhoG[-1], self.rhoL[-1], self.xTh[-1], self.Dhfg[-1], self.f[-1], self.areaMatrix_1[-1], self.areaMatrix_2[-1], self.areaMatrix, self.Vgj[-1], self.C0[-1], self.VgjPrime[-1]
        T_water_temp = self.T_water
        self.Ultemp, self.Ugtemp, self.Reltemp = self.Ul, self.Ug, self.Rel
        self.Ul, self.Ug, self.Rel = [], [], []
        self.P[-1], self.U[-1], self.H[-1], self.voidFraction[-1], self.rho[-1], self.rhoG[-1], self.rhoL[-1], self.xTh[-1], self.Dhfg[-1], self.f[-1], self.areaMatrix_1[-1], self.areaMatrix_2[-1], self.areaMatrix, self.Vgj[-1], self.C0[-1], self.VgjPrime[-1] = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        self.T_water = []
        for i in range(self.nCells):
            self.P[-1].append((Ptemp[i] + Ptemp[i+1])/2)
            self.U[-1].append((Utemp[i] + Utemp[i+1])/2)
            self.H[-1].append((Htemp[i] + Htemp[i+1])/2)
            self.voidFraction[-1].append((voidFractionTemp[i] + voidFractionTemp[i+1])/2)
            self.rho[-1].append((rhoTemp[i] + rhoTemp[i+1])/2)
            self.rhoG[-1].append((rhoGTemp[i] + rhoGTemp[i+1])/2)
            self.rhoL[-1].append((rhoLTemp[i] + rhoLTemp[i+1])/2)
            self.xTh[-1].append((xThTemp[i] + xThTemp[i+1])/2)
            self.Dhfg[-1].append((DhfgTemp[i] + DhfgTemp[i+1])/2)
            self.f[-1].append((fTemp[i] + fTemp[i+1])/2)
            self.areaMatrix_1[-1].append((areaMatrix_1Temp[i] + areaMatrix_1Temp[i+1])/2)
            self.areaMatrix_2[-1].append((areaMatrix_2Temp[i] + areaMatrix_2Temp[i+1])/2)
            self.areaMatrix.append((areaMatrixTemp[i] + areaMatrixTemp[i+1])/2)
            self.Vgj[-1].append((VgjTemp[i] + VgjTemp[i+1])/2)
            self.C0[-1].append((C0Temp[i] + C0Temp[i+1])/2)
            self.VgjPrime[-1].append((VgjPrimeTemp[i] + VgjPrimeTemp[i+1])/2)
            self.T_water.append((T_water_temp[i] + T_water_temp[i+1])/2)
            self.Ul.append((self.Ultemp[i] + self.Ultemp[i+1])/2)
            self.Ug.append((self.Ugtemp[i] + self.Ugtemp[i+1])/2)
            self.Rel.append((self.Reltemp[i] + self.Reltemp[i+1])/2)

        self.z_mesh = np.linspace(self.Dz/2, self.height-self.Dz/2, self.nCells)

    #Compute the surface temperature
    def compute_T_surf(self):
        self.Pfin = self.P[-1]
        self.h_z = self.H[-1]
        self.T_surf = np.zeros(self.nCells)
        self.Hc = np.zeros(self.nCells)
        self.Hnb = np.zeros(self.nCells)
        self.Hfc = np.zeros(self.nCells)
        updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.cladRadius, self.pin_pitch, self.pitch, self.D_h, self.areaMatrix, self.poro, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz, self.q__, self.phs, self.qFlow, self.fuelRadius, self.pitch/2, self.rsin_face, self.FAST_IAPWS)
        updateVariables.createFields()
        for i in range(self.nCells):
            hl, hg = updateVariables.getPhasesEnthalpy(i)
            P_MPa = self.Pfin[i]*10**-6 # [Mpa]
            Tsat = self.FAST_IAPWS.get_Tsat(P_MPa) # [K]
            DTSUB = updateVariables.getDTSUB(i)
            H_kJ = self.h_z[i]*10**-3
            phi = (self.q__[i] * self.areaMatrix[i])/(self.number_of_pins[i] * 2 * np.pi * self.cladRadius)
            rho_l = self.FAST_IAPWS.get_rhol(P_MPa)
            rho_g = self.FAST_IAPWS.get_rhog(P_MPa)
            mu_l = self.FAST_IAPWS.get_mul(P_MPa)
            mu_g = self.FAST_IAPWS.get_mug(P_MPa)
            k_l = self.FAST_IAPWS.get_kl(P_MPa)
            k_g = self.FAST_IAPWS.get_kg(P_MPa)
            C_l = self.FAST_IAPWS.get_cpl(P_MPa) * 1000.0
            C_g = self.FAST_IAPWS.get_cpg(P_MPa) * 1000.0
            sigma = self.FAST_IAPWS.get_sigma(P_MPa)
            h_fg = (hg - hl) * 1000.0
            # Liquid regime: Dittus-Boelter
            #Pr_l = (C_l * mu_l) / k_l
            Pr_g = (C_g * mu_g) / k_g
            Re_l_db = max(1e-4, updateVariables.getReynoldsNumberLiquid(i))
            #Hc_liq = 0.023 * (Pr_l**0.4) * (Re_l_db**0.8) * k_l / self.D_h[i]
            #T_surf_liq = (phi / Hc_liq) + self.T_water[i]
            # Liquid regime: Gnielinsky
            Pr_l = (C_l * mu_l) / k_l
            Re_l = max(1e-4, updateVariables.getReynoldsNumberLiquid(i))
            fric = updateVariables.getFrictionFactor(i)
            Nu_gnielinski = ((fric/8)*(Re_l-1000)*Pr_l)/(1 + 12.7*(fric/8)**(1/2)*(Pr_l**(2/3)-1))
            Hc_gnielinski = Nu_gnielinski * k_l / self.D_h[i]
            self.Hfc[i] = Hc_gnielinski
            T_surf_liq = (phi / Hc_gnielinski) + self.T_water[i]
            # Boiling regime: Chen
            G = self.rho[-1][i] * abs(self.U[-1][i])
            x_flow = updateVariables.getQuality(i)
            x_borne_sup = min_lisse(0.999, x_flow, 1e-4)
            x = max_lisse(1e-5, x_borne_sup, 1e-5)
            X_tt_inv = ((x/(1.0-x))**0.9) * ((rho_l/rho_g)**0.5) * ((mu_g/mu_l)**0.1)
            # Following conditions for the Reynolds Factor are taken from Groenveld and Snoek (10.1007/978-3-662-01657-2_3)
            if X_tt_inv > 0.1:
                F_calcul = 2.35 *(0.213 + X_tt_inv)**0.736
            else:
                F_calcul = 1
            
            F = if_lisse(X_tt_inv, 0.100207, 0.01, F_calcul, 1.0)
            Re_l_chen = max(1e-4, G *(1.0-x)*self.D_h[i]/mu_l)
            H_sp = 0.023 * (Pr_l**0.4)*(Re_l_chen**0.8)*k_l/self.D_h[i]
            H_c_chen = F * Hc_gnielinski #H_sp
            # S_fz = 1.0/(1.0 + 2.53e-6*(Re_l_chen*F**(1.25))**1.17) This correlation is defined by Orian et al. (10.1016/j.energy.2009.08.024)
            # for boiling flow in HORIZONTAL TUBES (CANDU reactors), therefore not adapted to BWR.
            
            # The following equations are taken from Groenveld and Snoek (10.1007/978-3-662-01657-2_3)
            Re_tp = F**(1.25)*Re_l_chen 
            if Re_tp < 32.5: # Re_tp < 32.5
                S_fz = 1/(1 + 0.12*Re_tp**(1.14))
            elif Re_tp < 70: # 32.5 =< Re_tp < 70
                S_fz = 1/(1 + 0.42*Re_tp**(0.78))
            else: # Re_tp > 70
                S_fz = 0.1

            C_fz = 0.00122 * (k_l**(0.79) * C_l**(0.45) * rho_l**(0.49))/(mu_l**(0.29) * sigma**(0.5) * h_fg**(0.24) * rho_g**0.24)
            T_surf_guess = self.T_water[i] + phi/H_c_chen
            #Newton loop for nucleate boiling
            for j in range(20):
                delta_T_sat = max_lisse(0.01, T_surf_guess - Tsat, 0.1)
                T_eval = min_lisse(T_surf_guess, 647.0, 0.5)
                
                try:
                    P_sat_wall = np.interp(T_eval, self.FAST_IAPWS.Tsat, self.FAST_IAPWS.P_arr) * 1e6
                except (NotImplementedError, ValueError):
                    P_sat_wall = P_MPa * 1e6
                    
                delta_P_sat = max_lisse(1.0, P_sat_wall - (P_MPa * 1e6), 10.0)
                
                H_fz = C_fz * (delta_T_sat**0.24) * (delta_P_sat**0.75)
                H_nb = S_fz * H_fz
                T_surf_new = (phi + H_nb * Tsat + H_c_chen * self.T_water[i]) / (H_nb + H_c_chen)
                
                if abs(T_surf_new - T_surf_guess) < 0.05:
                    T_surf_guess = T_surf_new
                    break
                T_surf_guess = 0.5 * T_surf_new + 0.5 * T_surf_guess
            self.Hnb[i] = H_fz
            T_surf_boil = T_surf_guess
            Hc_boil = H_nb + H_c_chen
            # Vapor regime: Dittus-Boelter
            Re_g_db = max(1e-4, updateVariables.getReynoldsNumberVapor(i))
            Hc_vap = 0.023 * (Pr_g**0.4) * (Re_g_db**0.8) * k_g / self.D_h[i]
            T_surf_vap = (phi / Hc_vap) + self.T_water
            #Global assembly
            T_onset = Tsat-DTSUB
            T_surf_trans1 = if_lisse(self.T_water[i], T_onset,0.005,T_surf_boil, T_surf_liq)
            Hc_trans1 = if_lisse(self.T_water[i], T_onset,0.005,Hc_boil,Hc_gnielinski) #Hc_liq)
            val_T = if_lisse(H_kJ, hg, 10.0, T_surf_vap, T_surf_trans1)
            val_Hc = if_lisse(H_kJ, hg, 10.0, Hc_vap, Hc_trans1)
            self.T_surf[i] = np.ravel(val_T)[0]
            self.Hc[i] = np.ravel(val_Hc)[0]
        return self.T_surf, self.Hfc, self.Hnb, self.Hc
    #Function to use the sous relaxation
    def sousRelaxation(self):

        for i in range(self.nFaces):
            self.voidFraction[-1][i] = self.voidFraction[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.voidFraction[-2][i]
            self.rho[-1][i] = self.rho[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rho[-2][i]
            self.rhoG[-1][i] = self.rhoG[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rhoG[-2][i]
            self.rhoL[-1][i] = self.rhoL[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rhoL[-2][i]
            self.xTh[-1][i] = self.xTh[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.xTh[-2][i]
            self.Vgj[-1][i] = self.Vgj[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.Vgj[-2][i]
            self.C0[-1][i] = self.C0[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.C0[-2][i]
            self.VgjPrime[-1][i] = self.VgjPrime[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.VgjPrime[-2][i]
    
    #Function to merge 2 lists into one
    def mergeVar(self, U, P):
        VAR = np.concatenate((U, P))
        return VAR
    
    #Function to split a list into 2
    def splitVar(self, VAR):
        U = VAR[:self.nFaces]
        P = VAR[self.nFaces:]
        return U, P
    
    #Function to get the phases velocity
    def getPhasesVelocity(self):
        water = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.cladRadius, self.pin_pitch, self.pitch, self.D_h, self.areaMatrix, self.poro, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz, self.q__, self.phs, self.qFlow, self.fuelRadius, self.pitch/2, self.rsin_face, self.FAST_IAPWS)
        Ul = [water.getUl(i) for i in range(self.nCells)]
        Ug = [water.getUg(i) for i in range(self.nCells)]
        return Ul, Ug
    
    #Function to get the phases saturation enthalpy
    def createBoundaryEnthalpy(self):
        for i in range(self.nFaces):
            self.hlSat.append(self.getPhasesEnthalpy(i)[0])
            self.hgSat.append(self.getPhasesEnthalpy(i)[1]) 
 
    #get the Reynolds number
    def getReynoldsNumber(self, i):
        mul = self.FAST_IAPWS.get_mul(self.P[-1][i]*10**-6)
        return (self.U[-1][i] * self.D_h[i] * self.rho[-1][i]) / mul
     
