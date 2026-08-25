from iapws import IAPWS97


Temperature = 287 + 273.15
Pressure = 7.2

Power = 870e6

steam = IAPWS97(T=Temperature, x=1)


enthaply = steam.h

print(enthaply)

print(Power / (enthaply * 1e3))