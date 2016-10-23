import AstroClass as AC
import pandas as pd
import numpy as np


def km2au(kilometers):
    return kilometers / 149597870.700


# test = AC.Body(1.192, 0.619, 70.8, 282.9, 171.4, 0.0, 'distance')
# test.set_epoch(2452700.5)
# stream = AC.Stream(test, 10000)

# np.savetxt('file.in', stream.get_info(), fmt='%17.9f', delimiter='\t')

earth = np.loadtxt('DATA/Evolut/closer/earth.out')
np.savetxt('DATA/Evolut/closer/earth.csv', earth, header='distance year jd planet node number', comments='')

planets = np.loadtxt('DATA/Evolut/closer/planets.out')
np.savetxt('DATA/Evolut/closer/planets.csv', planets, header='distance year jd planet number', comments='')

particles = np.loadtxt('DATA/Evolut/closer/file.out')
np.savetxt('DATA/Evolut/closer/particles.csv', particles, header='a e i Om w M year jd number node', comments='')


