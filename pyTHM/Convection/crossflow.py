import numpy as np

def compute_crossflow(DFM_actif, DFM_wr, hole_z_indices, hole_A, Idelchik_enter, Idelchik_exit, rwall_wr, v_lat_prev):
    """
    Calcule les transferts de masse, de quantité de mouvement et d'énergie entre l'écoulement actif et le water rod.
    (Version complète avec décalage de maillage Cellules -> Faces et corrélations physiques)
    """
    
    nCells = DFM_actif.nCells
    nFaces = DFM_actif.nFaces
    
    # Initialisation des vecteurs sources (qui seront injectés dans les matrices sur les FACES)
    S_mass_a = np.zeros(nFaces)
    S_mom_a  = np.zeros(nFaces)
    S_h_a    = np.zeros(nFaces)
    
    S_mass_w = np.zeros(nFaces)
    S_mom_w  = np.zeros(nFaces)
    S_h_w    = np.zeros(nFaces)
    
    # ---------------------------------------------------------
    # 1. TRANSFERT THERMIQUE PAR CONDUCTION (sur toute la hauteur)
    # ---------------------------------------------------------
    # La boucle se fait sur les volumes (cellules)
    for c in range(nCells):
        if rwall_wr[c] > 0:
            # Différence de température (T_wr - T_actif)
            delta_T = DFM_wr.T_water[c] - DFM_actif.T_water[c]
            
            # Q_cond = (Delta T / R_wall) * Dz [W]
            # Positif si la chaleur va du Water Rod vers l'Actif
            Q_cond = (delta_T / rwall_wr[c]) * DFM_actif.Dz
            
            # L'échange thermique calculé sur la cellule est affecté à la face sortante
            face_idx = c + 1
            S_h_a[face_idx] += Q_cond
            S_h_w[face_idx] -= Q_cond

    # ---------------------------------------------------------
    # 2. TRANSFERT DE MASSE ET ENTHALPIE (Écoulement par les trous)
    # ---------------------------------------------------------
    v_lat_new = np.zeros_like(v_lat_prev)
    
    # L'indice "c" est l'indice de la cellule où se trouve le trou
    for idx, c in enumerate(hole_z_indices):
        A_trou = hole_A[idx]
        if A_trou <= 0:
            continue
            
        # --- Récupération des variables locales au centre de la cellule "c" ---
        P_a, U_a, rho_l_a, eps_a, x_a, h_a = (
            DFM_actif.P[-1][c], DFM_actif.U[-1][c], DFM_actif.rhoL[-1][c], 
            DFM_actif.voidFraction[-1][c], DFM_actif.xTh[-1][c], DFM_actif.H[-1][c]
        )
        P_w, U_w, rho_l_w, eps_w, x_w, h_w = (
            DFM_wr.P[-1][c], DFM_wr.U[-1][c], DFM_wr.rhoL[-1][c], 
            DFM_wr.voidFraction[-1][c], DFM_wr.xTh[-1][c], DFM_wr.H[-1][c]
        )
        rho_g_a, rho_g_w = DFM_actif.rhoG[-1][c], DFM_wr.rhoG[-1][c]
        
        # Pression Totale Diphasique (Éq. issue de votre rapport)
        Ptot_a = P_a + 0.5 * (eps_a * rho_g_a * U_a**2 + (1 - eps_a) * rho_l_a * U_a**2)
        Ptot_w = P_w + 0.5 * (eps_w * rho_g_w * U_w**2 + (1 - eps_w) * rho_l_w * U_w**2)
        
        # --- Détermination du sens de l'écoulement (Amont -> Aval) ---
        # L'eau va de la Ptot la plus élevée vers la plus faible.
        if Ptot_a > Ptot_w:
            amont = 'actif'
            delta_P = P_a - P_w  # Différence de pression statique (utilisée par Idelchik)
            v_amont, v_aval = U_a, U_w
            rho_l_amont, eps_amont, x_amont, h_amont = rho_l_a, eps_a, x_a, h_a
        else:
            amont = 'wr'
            delta_P = P_w - P_a
            v_amont, v_aval = U_w, U_a
            rho_l_amont, eps_amont, x_amont, h_amont = rho_l_w, eps_w, x_w, h_w
            
        # --- Choix de la Table d'Idelchik ---
        # v1 est l'écoulement avec la plus petite vitesse
        v1_is_amont = (abs(v_amont) < abs(v_aval))
        
        # "Si v1 < v2: Si Ptot1 < Ptot2 -> Table 2 (Enter). Si Ptot1 > Ptot2 -> Table 3 (Exit)"
        if v1_is_amont:
            # v1 est en amont, donc Ptot1 est la pression haute (Ptot1 > Ptot2).
            table_data = np.array(Idelchik_exit)
        else:
            # v1 est en aval, donc Ptot1 est la pression basse (Ptot1 < Ptot2).
            table_data = np.array(Idelchik_enter)
            
        table_X = table_data[:, 0] 
        table_Y = table_data[:, 1]
            
        # --- Mini-Boucle locale de convergence pour Idelchik ---
        v_lat = v_lat_prev[idx] if v_lat_prev[idx] > 0.01 else 1.0 # Initial guess
        for _ in range(5): # 5 itérations suffisent généralement à converger
            ratio = abs(abs(v_aval) - abs(v_amont)) / v_lat
            # Interpolation dans les tables
            K_sing = np.interp(ratio, table_X, table_Y)
            
            # Éq 62 du rapport (en considérant phi_lo = 1.0 en première approche, modifiable si besoin)
            phi_lo2 = 1.0 
            
            # Calcul du débit massique (sécurité max(0, delta_P) pour éviter sqrt d'un négatif)
            m_dot = A_trou * np.sqrt( 2 * max(0, delta_P) * rho_l_amont / (phi_lo2 * K_sing) )
            
            # Mise à jour v_lat
            v_lat = m_dot / (rho_l_amont * A_trou)
            
        v_lat_new[idx] = v_lat
        
        # --- Corrélation de Saba & Lahey (Éq 64) ---
        n_SL = 3.0
        denom = x_amont + (1 - x_amont) * ((1 - eps_amont)**n_SL)
        x_lat = x_amont / denom if denom > 1e-5 else 0.0
        
        # --- Transferts finaux ---
        Q_lat = m_dot * h_amont     # [W] (Éq 65)
        Mom_lat = m_dot * v_amont   # [N] Transfert de quantité de mouvement
        
        # --- Affectation aux matrices sur la FACE ---
        face_idx = c + 1
        
        if amont == 'actif':
            # Sort de l'actif, entre dans le WR
            S_mass_a[face_idx] -= m_dot; S_h_a[face_idx] -= Q_lat; S_mom_a[face_idx] -= Mom_lat
            S_mass_w[face_idx] += m_dot; S_h_w[face_idx] += Q_lat; S_mom_w[face_idx] += Mom_lat
        else:
            # Sort du WR, entre dans l'actif
            S_mass_w[face_idx] -= m_dot; S_h_w[face_idx] -= Q_lat; S_mom_w[face_idx] -= Mom_lat
            S_mass_a[face_idx] += m_dot; S_h_a[face_idx] += Q_lat; S_mom_a[face_idx] += Mom_lat
            
    return S_mass_a, S_mom_a, S_h_a, S_mass_w, S_mom_w, S_h_w, v_lat_new