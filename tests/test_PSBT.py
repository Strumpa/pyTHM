# PSBT benchmark Phase I tests for pyTHM
import numpy as np
import pandas as pd
from pyTHM.Solver import pyTHM_solver
from conftest import (
    DATA_DIR,
    OUTPUTS_DIR
)

def get_axial_p_form(nz, height, shape):
    """
    Parameters:
    nz (int): Number of axial mesh points.
    height (float): Height of the fuel pin in cm.
    shape (str): Shape of the axial power distribution, can be "sine", "cosine", or "flat".
    """
    if shape == "sine":
        """
        get sine values over a mesh of nz points along the height h.
        """
        # Compute the boundaries and midpoints of each control volume
        z_boundaries = np.linspace(0, height, nz + 1)
        z_values = (z_boundaries[:-1] + z_boundaries[1:]) / 2  # Midpoints of control volumes
        # Compute sine values at each midpoint
        p_form_values = np.sin(np.pi * z_values / height) / np.mean(np.sin(np.pi * z_values / height))  # Normalize to have a mean of 1

    elif shape == "cosine":
        """
        get cosine values over a mesh of nz points along the height h.
        """
        # Compute the boundaries and midpoints of each control volume
        z_boundaries = np.linspace(0, height, nz + 1)
        z_values = (z_boundaries[:-1] + z_boundaries[1:]) / 2  # Midpoints of control volumes
        # Compute cosine values at each midpoint
        p_form_values = np.cos(np.pi * z_values / (2 * height)) / np.mean(np.cos(np.pi * z_values / (2 * height)))  # Normalize to have a mean of 1
    
    elif shape == "flat":
        """
        get flat values over a mesh of nz points along the height h.
        """
        p_form_values = np.ones(nz)
    else:
        raise ValueError("Shape must be 'sine', 'cosine', or 'flat'.")
    return p_form_values


# open PSBT data
## Recover PSBT data from cleaned CSV file
#----------
# PSBT DATA
# ---------
path_to_benchmark_data = DATA_DIR / 'PSBT' / 'PSBT_cleaned.csv'
df = pd.read_csv(path_to_benchmark_data, index_col=0)

# Extract the data
cols = df.columns[:7]

# Mettre chaque colonne dans une liste
lists = [df[col].tolist() for col in cols]

powerList = lists[3]
pressureList = lists[1]
temperatureList = lists[4]
flowList = lists[2]
test_id_numbers = lists[0]
densityResult = lists[5]
voidResult = lists[6]


nz = 70
zPlotting = [] #If empty, no plotting of the axial distribution of the fields, otherwise, list of the axial positions where the fields are plotted
## Meshing parameters:
If = 8
I1 = 3
height = 1.555  # in meters (converted from 155.5 cm)
shape = "flat"
pitch = 0.0126  # in meters (converted from 1.26 cm)
fuelRadius = 0.0027115493728018247  # in meters
inner_clad_radius = fuelRadius + 0.0000001  # in meters
outer_clad_radius = 0.0094996/2  # in meters
flow_cross_sectional_area = pitch**2 - np.pi * outer_clad_radius**2  # in square meters
hgap = 10000.0 # Convective heat transfer coefficient at gap in W/(m^2*K)
k_clad = 21.5  # thermal conductivity of clad in W/(m*K)
k_fuel = 4.18  # thermal conductivity of fuel in W/(m*K)   

PSBT_TEST_PARAMETERS = {}
for i in range(len(test_id_numbers)):
    test_id = str(test_id_numbers[i]) # Test ID number
    PSBT_TEST_PARAMETERS[test_id] = {}
    PSBT_TEST_PARAMETERS[test_id]["axial_p_form"] = get_axial_p_form(nz, height, shape)
    PSBT_TEST_PARAMETERS[test_id]["pOutlet"] = pressureList[i] * 98066.5 # Pa (convert kg/cm2a to Pa)
    PSBT_TEST_PARAMETERS[test_id]["tInlet"] = temperatureList[i] + 273.15 # K (convert °C to K)
    # Mass Flux is in 106kg/m2h, convert to kg/s: multiply by 10^6 to get actual value, divide by 3600 to convert h to s, multiply by area
    PSBT_TEST_PARAMETERS[test_id]["qFlow"] = flowList[i] * 1e6 / 3600 * flow_cross_sectional_area  # kg/s
    PSBT_TEST_PARAMETERS[test_id]["Power"] = powerList[i] * 1000  # Convert to W

def test_id_1p1222():
    test_id = "1.1222"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 452.07547479569655
    assert voidFrac[-1] == 0.2645615124210725
    assert P[0] - P[-1] == np.float64(16607042.828072965) - np.float64(16583403.79354423)

def test_id_1p1223():
    test_id = "1.1223"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 421.91336270229556
    assert voidFrac[-1] == 0.33022613315977556
    assert P[0] - P[-1] == np.float64(16614526.05733659) - np.float64(16583484.554753479)

def test_id_1p2211():
    test_id = "1.2211"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 420.0564373728348
    assert voidFrac[-1] == 0.3662539478713601
    assert P[0] - P[-1] == np.float64(14742747.640057564) - np.float64(14720205.859364692)

def test_id_1p2221():
    test_id = "1.2221"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 508.30853095536366
    assert voidFrac[-1] == 0.19477209607570267
    assert P[0] - P[-1] == np.float64(14736427.657541309) - np.float64(14720067.307139495)

def test_id_1p2223():
    test_id = "1.2223"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 398.1424525276732
    assert voidFrac[-1] == 0.40883467064412576
    assert P[0] - P[-1] == np.float64(14752340.547244819) - np.float64(14720285.66578096)

def test_id_1p2237():
    test_id = "1.2237"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 388.06852842643104
    assert voidFrac[-1] == 0.428180857205283
    assert P[0] - P[-1] == np.float64(14780266.605985072) - np.float64(14739953.548484087)

def test_id_1p2422():
    test_id = "1.2422"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 327.96592132597743
    assert voidFrac[-1] == 0.5451991503146474
    assert P[0] - P[-1] == np.float64(14731180.287866011) - np.float64(14719954.463042669)

def test_id_1p2423():
    test_id = "1.2423"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 300.3245745502728
    assert voidFrac[-1] == 0.5988613621937016
    assert P[0] - P[-1] == np.float64(14753462.670148812) - np.float64(14739606.991596445)

def test_id_1p4311():
    test_id = "1.4311"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 280.88969346103784
    assert voidFrac[-1] == 0.6442088856127333
    assert P[0] - P[-1] == np.float64(9859377.018958814) - np.float64(9846104.481661443)

def test_id_1p4312():
    test_id = "1.4312"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 242.17545213605806
    assert voidFrac[-1] == 0.7050296198787516
    assert P[0] - P[-1] == np.float64(9846125.517964859) - np.float64(9826590.203007437)


def test_id_1p4325():
    test_id = "1.4325"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 306.11415148016533
    assert voidFrac[-1] == 0.6046219813888065
    assert P[0] - P[-1] == np.float64(9850148.508971399) - np.float64(9836281.679510944)

def test_id_1p5221():
    test_id = "1.5221"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 394.9209628222452
    assert voidFrac[-1] == 0.4867585530739229
    assert P[0] - P[-1] == np.float64(7414426.684080841) - np.float64(7404172.847406715)

def test_id_1p5222():
    test_id = "1.5222"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 307.0132667986345
    assert voidFrac[-1] == 0.6137107824442605
    assert P[0] - P[-1] == np.float64(7370005.610670738) - np.float64(7355205.025676782)

def test_id_1p6221():
    test_id = "1.6221"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 365.69390919568445
    assert voidFrac[-1] == 0.5478104018172434
    assert P[0] - P[-1] == np.float64(4963731.594724134) - np.float64(4952535.645710561)

def test_id_1p6222():
    test_id = "1.6222"
    THsolve = pyTHM_solver(f"PSBT test {test_id}", "square", pitch, fuelRadius, inner_clad_radius, outer_clad_radius, 
                            height, 
                            PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            PSBT_TEST_PARAMETERS[test_id]["axial_p_form"], 
                            1.0, k_fuel, hgap, k_clad, nz, If, I1, zPlotting, 
                            solveConduction = True, 
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG")

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    assert len(Teff) == nz
    assert rho[-1] == 302.0181885130515
    assert voidFrac[-1] == 0.6326112382636541
    assert P[0] - P[-1] == np.float64(4917692.48753752) - np.float64(4903547.401716785)