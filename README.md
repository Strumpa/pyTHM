## pyTHM is an open-source python implementation of the THM: module of DONJON5
 - Stems from Clément Huet's Masters Thesis work : "Development and Validation of a Simplified Thermal-Hydraulics Solver for the Modelling of Boiling Water Reactor Fuel Channels in DONJON5", 2025,
 - Contributions from Marie Bellier : "Modélisation des écoulements diphasiques dans DONJON5" (in french) 2025.
 - Has served as a development prototype to test new implementations, in particular for the modelling of 2 phase flows in the context of Boiling Water Reactor modelling - "Coupled Neutronics and Thermal Hydraulics Simulations of a BWR Fuel Channel in the Version5 Environment" - R. Guasch, C. Huet, A. Hébert, C. Béguin and G. Marleau, presented at M&C2025, Denver, doi.org/10.13182/MC25-47128.
 - Current implementation is limited to a single channel drift flux model to solve the mass, momentum and energy conservation equations in the coolant.
 
 This version is meant as an external python package that can be coupled to the DONJON neutronics solvers in order to obtain coupled neutornics - thermalhydraulics solutions.

## To contribute :
Create your own branch, develop, make sure tests pass, submit a pull request to implement your changes.