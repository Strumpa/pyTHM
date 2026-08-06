import numpy as np
from pyTHM.Lissage.smooth import if_lisse, max_lisse, min_lisse
from pyTHM.WaterProperties.waterProperties import FAST_IAPWS

def compute_crossflow(DFM_actif, DFM_wr, hole_z_indices, hole_A, Idelchik_enter, Idelchik_exit, rwall_wr, v_lat_prev):
    """
    Calculates the mass, momentum, and enthalpy source terms for crossflow between the active channel and the water rod, considering both thermal conduction and mass transfer through holes.

    Attributes:
    - DFM_actif: Data structure containing the state of the active channel (pressure, velocity, density, void fraction, enthalpy, etc.).
    - DFM_wr: Data structure containing the state of the water rod (pressure, velocity, density, void fraction, enthalpy, etc.).
    - hole_z_indices: List of indices corresponding to the cells where holes are located.
    - hole_A: List of areas of the holes corresponding to the indices in hole_z_indices.
    - Idelchik_enter: Table of singular pressure drop coefficients for flow entering a hole (used for interpolation).
    - Idelchik_exit: Table of singular pressure drop coefficients for flow exiting a hole (used for interpolation).
    - rwall_wr: Array of water rod wall resistances for each cell (used for thermal conduction calculations).
    - v_lat_prev: Array of previous lateral velocities for each hole (used for iterative convergence).
    """
    
    nCells = DFM_actif.nCells
    nFaces = DFM_actif.nFaces
    
    # Initialization of source vectors (which will be injected into the matrices on the FACES)
    S_mass_a = np.zeros(nFaces)
    S_mom_a  = np.zeros(nFaces)
    S_h_a    = np.zeros(nFaces)
    
    S_mass_w = np.zeros(nFaces)
    S_mom_w  = np.zeros(nFaces)
    S_h_w    = np.zeros(nFaces)
    
    # ---------------------------------------------------------
    # 1. THERMAL HEAT TRANSFER BY CONDUCTION (over the full height)
    # ---------------------------------------------------------
    # The loop iterates over volumes (cells)
    for c in range(nCells):
        if rwall_wr[c] > 0:
            P_a_MPa = DFM_actif.P[-1][c] * 1e-6
            P_w_MPa = DFM_wr.P[-1][c] * 1e-6
            
            # --- 1. Convective coefficient of the Active channel (Dittus-Boelter) ---
            mu_a = FAST_IAPWS.get_mul(P_a_MPa)
            cp_a = FAST_IAPWS.get_cpl(P_a_MPa) * 1000.0
            k_a = FAST_IAPWS.get_kl(P_a_MPa)
            
            G_a = DFM_actif.rho[-1][c] * abs(DFM_actif.U[-1][c])
            Re_a = max(1e-4, G_a * DFM_actif.D_h[c] / mu_a)
            Pr_a = (cp_a * mu_a) / k_a
            
            h_a = (0.023 * (Re_a**0.8) * (Pr_a**0.4) * k_a) / DFM_actif.D_h[c]
            
            # --- 2. Convective coefficient of the Water Rod (Dittus-Boelter) ---
            mu_w = FAST_IAPWS.get_mul(P_w_MPa)
            cp_w = FAST_IAPWS.get_cpl(P_w_MPa) * 1000.0
            k_w = FAST_IAPWS.get_kl(P_w_MPa)
            
            G_w = DFM_wr.rho[-1][c] * abs(DFM_wr.U[-1][c])
            Re_w = max(1e-4, G_w * DFM_wr.D_h[c] / mu_w)
            Pr_w = (cp_w * mu_w) / k_w
            
            h_w = (0.023 * (Re_w**0.8) * (Pr_w**0.4) * k_w) / DFM_wr.D_h[c]
            
            # --- 3. Thermal Resistance Balance ---
            # Using the hydraulic diameter of the WR for the tube perimeter (pi * D)
            perim_wr = np.pi * DFM_wr.D_h[c]
            
            # Linear resistances in K.m / W
            R_conv_a = 1.0 / (h_a * perim_wr)
            R_conv_w = 1.0 / (h_w * perim_wr)
            R_paroi = rwall_wr[c]
            
            # Sum of resistances in series
            R_tot = R_paroi + R_conv_a + R_conv_w
            
            # --- 4. Heat Flux Calculation ---
            T_a = FAST_IAPWS.get_sub_T(P_a_MPa, DFM_actif.H[-1][c] * 1e-3)
            T_w = FAST_IAPWS.get_sub_T(P_w_MPa, DFM_wr.H[-1][c] * 1e-3)
            
            delta_T = T_w - T_a
            Q_cond = (delta_T / R_tot) * DFM_actif.Dz
            
            # Assignment to the outgoing face
            face_idx = c + 1
            S_h_a[face_idx] += Q_cond
            S_h_w[face_idx] -= Q_cond

    # ---------------------------------------------------------
    # 2. MASS AND ENTHALPY TRANSFER (Flow through holes)
    # ---------------------------------------------------------
    v_lat_new = np.zeros_like(v_lat_prev)
    
    # Index "c" is the index of the cell where the hole is located
    for idx, c in enumerate(hole_z_indices):
        A_trou = hole_A[idx]
        if A_trou <= 0:
            continue
            
        # --- Retrieval of local variables at the center of cell "c" ---
        P_a, U_a, rho_l_a, eps_a, x_a, h_a = (
            DFM_actif.P[-1][c], DFM_actif.U[-1][c], DFM_actif.rhoL[-1][c], 
            DFM_actif.voidFraction[-1][c], DFM_actif.xTh[-1][c], DFM_actif.H[-1][c]
        )
        P_w, U_w, rho_l_w, eps_w, x_w, h_w = (
            DFM_wr.P[-1][c], DFM_wr.U[-1][c], DFM_wr.rhoL[-1][c], 
            DFM_wr.voidFraction[-1][c], DFM_wr.xTh[-1][c], DFM_wr.H[-1][c]
        )
        rho_g_a, rho_g_w = DFM_actif.rhoG[-1][c], DFM_wr.rhoG[-1][c]
        
        # Total Two-Phase Pressure
        Ptot_a = P_a + 0.5 * (eps_a * rho_g_a * U_a**2 + (1 - eps_a) * rho_l_a * U_a**2)
        Ptot_w = P_w + 0.5 * (eps_w * rho_g_w * U_w**2 + (1 - eps_w) * rho_l_w * U_w**2)
        
        # --- Determination of flow direction (Upstream -> Downstream) ---
        # Water flows from the highest Ptot to the lowest.
        if Ptot_a > Ptot_w:
            amont = 'actif'
            delta_P = P_a - P_w  # Static pressure difference (used by Idelchik)
            v_amont, v_aval = U_a, U_w
            rho_l_amont, eps_amont, x_amont, h_amont = rho_l_a, eps_a, x_a, h_a
        else:
            amont = 'wr'
            delta_P = P_w - P_a
            v_amont, v_aval = U_w, U_a
            rho_l_amont, eps_amont, x_amont, h_amont = rho_l_w, eps_w, x_w, h_w
            
        # --- Choice of Idelchik Table ---
        # v1 is the flow with the smallest velocity
        v1_is_amont = (abs(v_amont) < abs(v_aval))
        
        # "If v1 < v2: If Ptot1 < Ptot2 -> Table 2 (Enter). If Ptot1 > Ptot2 -> Table 3 (Exit)"
        if v1_is_amont:
            # v1 is upstream, so Ptot1 is the high pressure (Ptot1 > Ptot2).
            table_data = np.array(Idelchik_exit)
        else:
            # v1 is downstream, so Ptot1 is the low pressure (Ptot1 < Ptot2).
            table_data = np.array(Idelchik_enter)
            
        table_X = table_data[:, 0] 
        table_Y = table_data[:, 1]
            
        # --- Local mini-convergence loop for Idelchik ---
        v_lat = v_lat_prev[idx] if v_lat_prev[idx] > 0.01 else 1.0 # Initial guess
        for _ in range(5): # 5 iterations are generally sufficient to converge
            ratio = abs(abs(v_aval) - abs(v_amont)) / max(abs(v_lat),1e-10)
            # Interpolation in the tables
            K_sing = np.interp(ratio, table_X, table_Y)
            
            # Eq. 62 from the report (assuming phi_lo = 1.0 as a first approximation, adjustable if needed)
            phi_lo2 = 1.0 
            
            # Mass flow rate calculation (safety: max(0, delta_P) to avoid sqrt of a negative number)
            delta_P_lisse = max_lisse(1e-4, delta_P, 10.0)
            m_dot = A_trou * np.sqrt( 2 * max(0, delta_P_lisse) * rho_l_amont / (phi_lo2 * K_sing) )
            
            # Update v_lat
            v_lat = m_dot / (rho_l_amont * A_trou)
            
        v_lat_new[idx] = v_lat
        
        # --- Saba & Lahey correlation (Eq. 64) ---
        n_SL = 3.0
        denom = x_amont + (1 - x_amont) * ((1 - eps_amont)**n_SL)
        x_lat = x_amont / denom if denom > 1e-5 else 0.0
        
        # --- Final transfers ---
        Q_lat = m_dot * h_amont     # [W] (Eq. 65)
        Mom_lat = m_dot * v_amont   # [N] Momentum transfer
        
        # --- Assignment to matrices on the FACE ---
        face_idx = c + 1
        
        if amont == 'actif':
            # Exits the active channel, enters the WR
            S_mass_a[face_idx] -= m_dot; S_h_a[face_idx] -= Q_lat; S_mom_a[face_idx] -= Mom_lat
            S_mass_w[face_idx] += m_dot; S_h_w[face_idx] += Q_lat; S_mom_w[face_idx] += Mom_lat
        else:
            # Exits the WR, enters the active channel
            S_mass_w[face_idx] -= m_dot; S_h_w[face_idx] -= Q_lat; S_mom_w[face_idx] -= Mom_lat
            S_mass_a[face_idx] += m_dot; S_h_a[face_idx] += Q_lat; S_mom_a[face_idx] += Mom_lat
            
    return S_mass_a, S_mom_a, S_h_a, S_mass_w, S_mom_w, S_h_w, v_lat_new