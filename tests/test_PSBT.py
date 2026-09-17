# PSBT benchmark Phase I tests for pyTHM
import numpy as np
import os
import pandas as pd
import pytest
from pyTHM.Solver.main import pyTHM_solver
from pyTHM.WaterProperties import FastIAPWS
from conftest import (
   PSBT_DATA,
    OUTPUTS_DIR
)


def get_z_mesh(h, nz):
    """
    Returns a list of z mesh points based on the height and number of axial mesh points.
    """
    # Compute the boundaries and midpoints of each control volume
    z_boundaries = np.linspace(0, h, nz + 1)
    z_values = (z_boundaries[:-1] + z_boundaries[1:]) / 2  # Midpoints of control volumes
    return z_values, z_boundaries

def interpolate_void_fraction(zmesh, voidfractions):
    PSBT_measurement_height = 1.40
    vf_at_140cm = np.interp(PSBT_measurement_height, zmesh, voidfractions)
    return vf_at_140cm

def get_PSBT_geometric_data(nz, pitch, cladRadius):

    geometric_data = {}
    geometric_data["active_flow_data"] = {}
    geometric_data["fuel_data"] = {}
    acools = np.array(nz*[pitch**2 - np.pi*cladRadius**2])
    porosities = acools / pitch**2 #np.array(nz*[1])#
    wetted_perimeters = np.array(nz*[2*np.pi*cladRadius])
    hydraulic_diameters = 4 * acools / wetted_perimeters
    heated_perimeters = np.array(nz*[2*np.pi*cladRadius])

    geometric_data["fuel_data"]["fuel_radius"] = fuelRadius # fuel pin radius in meters
    geometric_data["fuel_data"]["gap_radius"] = inner_clad_radius# gap radius in meters, used to determine mesh elements for constant surface discretization
    geometric_data["fuel_data"]["clad_radius"] = cladRadius# clad radius in meters, used to determine mesh elements for constant surface discretization
    geometric_data["fuel_data"]["pin_pitch"] = pitch # distance between the centers of two adjacent fuel rods in meters
    geometric_data["fuel_data"]["max_rod_length"] = height

    geometric_data["active_flow_data"]["number_of_axial_meshes"] = nz
    geometric_data["active_flow_data"]["porosities"] = porosities
    geometric_data["active_flow_data"]["coolant_cross_sectional_areas"] = acools
    geometric_data["active_flow_data"]["hydraulic_diameters"] = hydraulic_diameters
    geometric_data["active_flow_data"]["heated_perimeters"] = heated_perimeters
    geometric_data["active_flow_data"]["k_expansion"] = np.array(nz * [0.0])
    geometric_data["active_flow_data"]["k_contraction"]  = np.array(nz * [0.0])
    geometric_data["active_flow_data"]["singular_contraction_ratios"] = np.array(nz * [1.0])
    geometric_data["active_flow_data"]["pitch"] = pitch

    return geometric_data


# open PSBT data
## Recover PSBT data from cleaned CSV file
#----------
# PSBT DATA
# ---------
path_to_benchmark_data = PSBT_DATA + '/PSBT_cleaned.csv'
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
pitch = 0.0126  # in meters (converted from 1.26 cm)
fuelRadius = 0.0027115493728018247  # in meters
inner_clad_radius = fuelRadius + 0.0000001  # in meters
outer_clad_radius = 0.0094996/2  # in meters
flow_cross_sectional_area = pitch**2 - np.pi * outer_clad_radius**2  # in square meters
hgap = 10000.0 # Convective heat transfer coefficient at gap in W/(m^2*K)
k_clad = 21.5  # thermal conductivity of clad in W/(m*K)
k_fuel = 4.18  # thermal conductivity of fuel in W/(m*K)   

zmesh_centers, zmesh_boundaries = get_z_mesh(height, nz)

geometric_data = get_PSBT_geometric_data(nz, pitch, outer_clad_radius)

PSBT_TEST_PARAMETERS = {}
pressures = []
for i in range(len(test_id_numbers)):
    test_id = str(test_id_numbers[i]) # Test ID number
    print(f"PSBT test ID = {test_id}")
    PSBT_TEST_PARAMETERS[test_id] = {}
    PSBT_TEST_PARAMETERS[test_id]["power_shape"] = "uniform" # All tests have uniform power distribution
    PSBT_TEST_PARAMETERS[test_id]["pOutlet"] = pressureList[i] * 98066.5 # Pa (convert kg/cm2a to Pa)
    print(f"Pressure={pressureList[i] * 98066.5} MPa")
    pressures.append(pressureList[i] * 98066.5)
    PSBT_TEST_PARAMETERS[test_id]["tInlet"] = temperatureList[i] + 273.15 # K (convert °C to K)
    # Mass Flux is in 10^6kg/m2h, convert to kg/s: multiply by 10^6 to get actual value, divide by 3600 to convert h to s, multiply by area
    PSBT_TEST_PARAMETERS[test_id]["qFlow"] = flowList[i] * 1e6 / 3600 * flow_cross_sectional_area  # kg/s
    PSBT_TEST_PARAMETERS[test_id]["Power"] = powerList[i] * 1000  # Convert to W
    PSBT_TEST_PARAMETERS[test_id]["vF"] = voidResult[i]

tables = {}
table16 = FastIAPWS(P_outlet=16.58e6)
table14 = FastIAPWS(P_outlet=14.71e6)
#table12 = FastIAPWS(P_outlet=12.25e6)
table9 = FastIAPWS(P_outlet=9.78e6)
table7 = FastIAPWS(P_outlet=7.35e6)
table4 = FastIAPWS(P_outlet=4.90e6)

tables["1.1222"] = table16
tables["1.1223"] = table16
tables["1.2211"] = table14
tables["1.2221"] = table14
tables["1.2223"] = table14
tables["1.2237"] = table14
tables["1.2422"] = table14
tables["1.2423"] = table14
tables["1.4311"] = table9
tables["1.4312"] = table9
tables["1.4325"] = table9
tables["1.5221"] = table7
tables["1.5222"] = table7
tables["1.6221"] = table4
tables["1.6222"] = table4


def test_id_1p1222():
    test_id = "1.1222"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                            channel_type="square", 
                            geometric_data=geometric_data,
                            tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                            pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                            qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                            Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                            power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                            fraction_pow_fuel=1.0, 
                            k_fuel=k_fuel, 
                            H_gap=hgap, 
                            k_clad=k_clad, 
                            I_f=If, 
                            I_c=I1, 
                            plot_at_z=zPlotting, 
                            solveConduction = True, 
                            fast_iapws_table=tables[test_id],
                            dt = 0, t_tot = 0, 
                            frfaccorel = 'Churchill', 
                            P2Pcorel = 'lockhartMartinelli', 
                            voidFractionCorrel = 'EPRIvoidModel', 
                            numericalMethod = "BiCG",
                        )

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(503.92, abs=1e-2)
    assert deltaP == pytest.approx(31726, abs=1)

def test_id_1p1223():
    test_id = "1.1223"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad,  
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(458.69, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.31068, abs=1e-5)
    assert deltaP == pytest.approx(53525, abs=1)

def test_id_1p2211():
    test_id = "1.2211"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(502.98, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.20301, abs=1e-5)
    assert deltaP == pytest.approx(25963, abs=1)

def test_id_1p2221():
    test_id = "1.2221"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(610.02, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.0, abs=1e-5)
    assert deltaP == pytest.approx(17935, abs=1)

def test_id_1p2223():
    test_id = "1.2223"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(423.83, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.35725, abs=1e-5)
    assert deltaP == pytest.approx(57930, abs=1)

def test_id_1p2237():
    test_id = "1.2237"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(399.15, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.40653, abs=1e-5)
    assert deltaP == pytest.approx(88403, abs=1)

def test_id_1p2422():
    test_id = "1.2422"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(397.51, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.40848, abs=1e-5)
    assert deltaP == pytest.approx(16997, abs=1)

def test_id_1p2423():
    test_id = "1.2423"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(339.17, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.52317, abs=1e-5)
    assert deltaP == pytest.approx(31172, abs=1)

def test_id_1p4311():
    test_id = "1.4311"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.20)
    assert rho[-1] == pytest.approx(361.14, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.51573, abs=1e-5)
    assert deltaP == pytest.approx(19869, abs=1)

def test_id_1p4312():
    test_id = "1.4312"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",
                            )
    

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(271.68, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.65791, abs=1e-5)
    assert deltaP == pytest.approx(59122, abs=1)


def test_id_1p4325():
    test_id = "1.4325"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",                              
                            )

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(345.61, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.54097, abs=1e-5)
    assert deltaP == pytest.approx(26685, abs=1)

def test_id_1p5221():
    test_id = "1.5221"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG"                            
                            )

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(494.78, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.33922, abs=1e-5)
    assert deltaP == pytest.approx(11377, abs=1)

def test_id_1p5222():
    test_id = "1.5222"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",                               
                            )

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.15)
    assert rho[-1] == pytest.approx(329.49, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.57822, abs=1e-5)
    assert deltaP == pytest.approx(30690, abs=1)

def test_id_1p6221():
    test_id = "1.6221"
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",                         
                            )

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.20)
    assert rho[-1] == pytest.approx(456.07, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.42584, abs=1e-5)
    assert deltaP == pytest.approx(12528, abs=1)

def test_id_1p6222():
    test_id = "1.6222"
    print(f"POutlet = {PSBT_TEST_PARAMETERS[test_id]['pOutlet']}")
    print(f"tInlet = {PSBT_TEST_PARAMETERS[test_id]['tInlet']}")
    THsolve = pyTHM_solver( case_name=f"PSBT test {test_id}", 
                                channel_type="square", 
                                geometric_data=geometric_data,
                                tInlet=PSBT_TEST_PARAMETERS[test_id]["tInlet"], 
                                pOutlet=PSBT_TEST_PARAMETERS[test_id]["pOutlet"], 
                                qFlow=PSBT_TEST_PARAMETERS[test_id]["qFlow"], 
                                Powtot=PSBT_TEST_PARAMETERS[test_id]["Power"], 
                                power_distribution=PSBT_TEST_PARAMETERS[test_id]["power_shape"], 
                                fraction_pow_fuel=1.0, 
                                k_fuel=k_fuel, 
                                H_gap=hgap, 
                                k_clad=k_clad, 
                                I_f=If, 
                                I_c=I1, 
                                plot_at_z=zPlotting, 
                                solveConduction = True, 
                                fast_iapws_table=tables[test_id],
                                dt = 0, t_tot = 0, 
                                frfaccorel = 'Churchill', 
                                P2Pcorel = 'lockhartMartinelli', 
                                voidFractionCorrel = 'EPRIvoidModel', 
                                numericalMethod = "BiCG",                            
                            )

    Teff, Twater, rho, voidFrac, P, U, H = THsolve.get_TH_parameters()
    deltaP = P[0] - P[-1]
    assert len(Teff) == nz
    assert interpolate_void_fraction(zmesh_centers,voidFrac) == pytest.approx(PSBT_TEST_PARAMETERS[test_id]["vF"], abs=0.22)
    assert rho[-1] == pytest.approx(334.56, abs=1e-2)
    assert voidFrac[-1] == pytest.approx(0.58793, abs=1e-5)
    assert deltaP == pytest.approx(23991, abs=1)