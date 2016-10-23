import pandas as pd
import matplotlib


matplotlib.style.use('ggplot')

de = pd.read_csv('deaths.csv')
ve = de[de.planet == 2]
ea = de[de.planet == 3]
ju = de[de.planet == 5]
ax = ve.plot.scatter('year', 'distance', color='DarkGreen', label='Venus', s=70)
bx = ea.plot.scatter('year', 'distance', color='DarkBlue', label='Earth', ax=ax, s=70)
cx = ju.plot.scatter('year', 'distance', color='DarkRed', label='Jupiter', ax=bx, s=70)
ax = coor.plot('year', 'r', color='DarkGreen')
bx = pl.plot.scatter('year', 'distance', color='DarkBlue', s=30, ax=ax)
cx = jj.plot('year', 'r', color='DarkRed', label='Jupiter', ax=bx)