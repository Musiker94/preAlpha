from astropy.time import Time
import numpy as np
import datetime
import random
import shutil
import os


__author__ = 'KMS 2016'


def mu(dim='solar system'):
    """Return the value of gravitational constant"""
    dictionary = {'solar system': 2.9591220828559110225E-4,
                  'earth': 3.9864418E05}
    # mu = 2.9591220828559110225E-4  # Gravitational constant = [au^3/day^2]
    return dictionary[dim]


def julian(year, month, day, hours, minutes, seconds):
    """Return julian date"""
    if not isinstance(seconds, int):
        microseconds = seconds - int(seconds)
        microseconds *= 10**6
        date = datetime.datetime(year, month, day, hours, minutes, int(seconds), int(microseconds))
    else:
        date = datetime.datetime(year, month, day, hours, minutes, seconds)

    iso = ['']      # Astropy works with arrays
    iso[0] = date.isoformat()
    time = Time(iso, format='isot', scale='utc')
    return time.jd[0]


def get_time(year, month, day, hours, minutes, seconds):
    """Return Time object from Astropy"""
    if not isinstance(seconds, int):
        microseconds = seconds - int(seconds)
        microseconds *= 10**6
        date = datetime.datetime(year, month, day, hours, minutes, int(seconds), int(microseconds))
    else:
        date = datetime.datetime(year, month, day, hours, minutes, seconds)

    iso = ['']      # Astropy works with arrays
    iso[0] = date.isoformat()
    time = Time(iso, format='isot', scale='utc')
    return time
