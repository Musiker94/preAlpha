import pandas as pd


planets = pd.read_csv('planets.csv', ' ')
now = planets[planets.number == 1]
avg = sum(now.distance) / len(now)
now = now[now.distance <= avg]

