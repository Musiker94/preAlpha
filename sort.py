import pandas as pd


planets = pd.read_csv('planets.csv', ' ')
unique = planets.number.unique()
deaths = planets[planets.number == unique[0]]
deaths = deaths[deaths.index == deaths.first_valid_index()]

for count in unique[1:]:
    now = planets[planets.number == count]
    now = now[now.index == now.first_valid_index()]
    deaths = deaths.append(now)

deaths.to_csv('deaths.csv', index_label='ID')
