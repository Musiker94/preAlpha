import AstroClass
import pandas as pd
import numpy as np


closer = pd.read_csv('DATA/Evolut/closer_2675to3000/particles.csv', ' ')
coordinates = np.zeros([len(closer), 8])
for count in range(0, len(closer)):
    elem = closer[closer.index == count].get_values()[0]
    elem[2:6] = np.radians(elem[2:6])
    elem[5] = AstroClass.mean2true_anomaly(elem[0], elem[1], elem[5])    # procedure uses TRUE ANOMALY!!
    coordinates[count, 0:3] = AstroClass.kepler2descart(elem[0:6], dim='rad')[0]
    coordinates[count, 3:6] = AstroClass.kepler2descart(elem[0:6], 'rad')[1]
    coordinates[count, 6] = np.sqrt(np.dot(coordinates[count][0:3], coordinates[count][0:3]))
    coordinates[count, 7] = elem[6]
    print('Count = %i' % count)

np.savetxt('DATA/Evolut/closer_2675to3000/coordinates.csv', coordinates, header='X Y Z velX velY velZ R year', comments='')

#
# jup = AstroClass.Body(5.204267, 0.048775, 1.03, 100.55615, 275.066, 0.0)
# jup.set_date(2010, 10, 10, 0, 0, 0)
# approx = np.zeros([360, 9])
# n = np.sqrt(AstroClass.MU / jup.get_orbit()[0] ** 3)
# for deg in range(0, 360):
#     jup.set_anomaly(deg)
#     approx[deg, 0:3] = jup.get_position()[0]
#     approx[deg, 3:6] = jup.get_position()[1]
#     approx[deg, 6] = jup.get_orbit(anomaly='mean')[5]
#     approx[deg, 8] = np.sqrt(np.dot(approx[deg, 0:3], approx[deg, 0:3]))
#
# if approx[deg, 6] >= np.pi:
#     approx[deg, 7] = jup.get_date('jd') - approx[deg, 6] / n
# else:
#     approx[deg, 7] = jup.get_date('jd') + approx[deg, 6] / n
# jup.set_epoch(approx[deg, 7])
# approx[deg, 7] = jup.get_date('byear')
