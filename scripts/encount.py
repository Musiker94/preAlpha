"""
df - Data Frame for pandas

"""
import AstroClass
import pandas as pd
import numpy as np


path_closer = r'DATA/Evolut/closer/'
path_closer23 = r'DATA/Evolut/closer_2675to3000/'

df_particles = pd.read_csv(path_closer23 + 'particles.csv', ' ')
df_planets = pd.read_csv(path_closer + 'planets.csv', ' ')

indexes = AstroClass.local_min(df_planets.distance.get_values())
year = df_planets.get_value(df_planets.distance.idxmin(), 'year')

