from astropy.time import Time
import numpy as np
import datetime
import random
import math


def mu(dim='solar system'):
    """Returns the value of gravitational constant - dimension for 'solar system' [au^3/day^2] (default)"""
    dictionary = {'solar system': 2.9591220828559110225E-4,
                  'earth': 3.9864418E05}
    return dictionary[dim]

MU = mu()


def kepler(eccentricity, mean_anomaly):
    """Solve Kepler's eq"""
    cos = eccentricity * math.cos(mean_anomaly)
    sin = eccentricity * math.sin(mean_anomaly)
    psi = sin / math.sqrt(1.0 - cos - cos + eccentricity * eccentricity)
    while True:
        xi = math.cos(psi)
        eta = math.sin(psi)
        fd = (1.0 - cos * xi) + sin * eta
        fdd = cos * eta + sin * xi
        f = psi - fdd
        psi -= f * fd / (fd * fd - 0.5 * f * fdd)
        if f * f <= 1.0e-12:
            break
    return mean_anomaly + psi


def mean2true_anomaly(axis, eccentricity, mean_anomaly):
    """Converts the mean anomaly to the true anomaly. Dimension in radians!"""
    ee = kepler(eccentricity, mean_anomaly)
    r = axis * (1.0 - eccentricity * math.cos(ee))
    sv = axis * math.sqrt(1.0 - eccentricity * eccentricity) * math.sin(ee) / r
    cv = axis * (math.cos(ee) - eccentricity) / r
    true_anomaly = math.atan2(sv, cv)
    if true_anomaly < 0:
        true_anomaly += 2 * math.pi
    elif true_anomaly >= 2 * math.pi:
        true_anomaly -= 2 * math.pi
    return true_anomaly


def true2mean_anomaly(eccentricity, true_anomaly):
    """Converts the true anomaly to the mean anomaly. Dimension in radians!"""
    e = np.tan(true_anomaly / 2.0) * np.sqrt((1.0 - eccentricity) / (1.0 + eccentricity))
    e = np.arctan(e) * 2.0
    return e - eccentricity * np.sin(e)


def descart2kepler(coordinates, velocities):
    """Returns Keplerian elements

    :param coordinates: vector of coordinates of the body
    :param velocities: vector of velocities of the body
    :return:a - semimajor axis,
            e - eccentricity,
            i - inclination,
            om - ascending node,
            w - argument perihelion,
            v - true anomaly
            (all angular values in radians)

    """
    r = math.sqrt(np.dot(coordinates, coordinates))
    sigma3 = np.cross(coordinates, velocities)
    sigma = math.sqrt(np.dot(sigma3, sigma3))
    cos_i = sigma3[2] / sigma
    sin_i = math.sqrt(1 - math.pow(cos_i, 2))
    i = np.arctan2(sin_i, cos_i)
    sin_om = sigma3[0] / (sigma * sin_i)
    cos_om = - sigma3[1] / (sigma * sin_i)
    om = np.arctan2(sin_om, cos_om)
    par = math.pow(sigma, 2) / MU
    h = np.dot(velocities, velocities) - 2 * MU / r
    e = 1 + h * math.pow(sigma, 2) / math.pow(MU, 2)  # eccentricity^2
    a = par / (1 - e)
    e = math.sqrt(e)
    tg_v = np.dot(coordinates, velocities) * math.sqrt(par / MU) / (par - r)
    cos_v = (par - r) / (r * e)
    sin_v = tg_v * cos_v
    v = np.arctan2(sin_v, cos_v)
    tg_u = coordinates[2] / (sin_i * (coordinates[0] * cos_om + coordinates[1] * sin_om))
    sin_u = coordinates[2] / (r * sin_i)
    cos_u = sin_u / tg_u
    u = np.arctan2(sin_u, cos_u)
    if v != 0.0:
        w = u - v + 2 * math.pi
    else:
        w = u

    if w >= 2 * math.pi:
        w -= 2 * math.pi

    if om < 0:
        om += 2 * math.pi
    elif om >= 2 * math.pi:
        om -= 2 * math.pi

    if v < 0:
        v += 2 * math.pi
    elif v >= 2 * math.pi:
        v -= 2 * math.pi

    return a, e, i, om, w, v


def kepler2descart(elements, dim='deg', out_type='sep'):
    """Returns coordinates and velocities of the body

    :param elements: vector of Keplerian elements:
                    [semi major axis,
                    eccentricity,
                    inclination,
                    ascending node,
                    argument of perihelion,
                    true anomaly]
    :param dim: sets the dimension of the angular values:
                'deg' - degrees (default),
                'rad' - radians
    :param out_type: sets the type of output values:
                'sep' - out two vectors: "coordinates" and "velocities"
                'uni' - out one vector of six parameters
    :return: vector of coordinates and vector of velocities

    """
    axis = elements[0]
    eccentricity = elements[1]
    inclination = elements[2]
    node = elements[3]
    perihelion = elements[4]
    true_anomaly = elements[5]
    if dim == 'deg':
        inclination = math.radians(inclination)
        node = math.radians(node)
        perihelion = math.radians(perihelion)
        true_anomaly = math.radians(true_anomaly)

    u = true_anomaly + perihelion
    parameter = axis * (1.0 - eccentricity * eccentricity)
    parameter_div_radius = 1 + eccentricity * math.cos(true_anomaly)
    radius_vector = parameter / parameter_div_radius
    n = math.sqrt(MU / parameter)
    vr = n * eccentricity * math.sin(true_anomaly)
    vn = n * parameter_div_radius
    # v_sqr = MU * (2 * elements[0] - radius_vector) / (radius_vector * elements[0])
    p = []
    q = []
    coordinates = []
    velocities = []

    cos_u = math.cos(u)
    sin_u = math.sin(u)
    cos_om = math.cos(node)
    sin_om = math.sin(node)
    cos_i = math.cos(inclination)
    sin_i = math.sin(inclination)

    p.append(cos_u * cos_om - sin_u * sin_om * cos_i)
    p.append(cos_u * sin_om + sin_u * cos_om * cos_i)
    p.append(sin_u * sin_i)
    q.append(-sin_u * cos_om - cos_u * sin_om * cos_i)
    q.append(-sin_u * sin_om + cos_u * cos_om * cos_i)
    q.append(cos_u * sin_i)

    for count in range(0, 3):
        coordinates.append(radius_vector * p[count])

    for count in range(0, 3):
        velocities.append(p[count] * vr + q[count] * vn)

    if out_type == 'sep':
        return coordinates, velocities
    elif out_type == 'uni':
        out = np.zeros(6)
        for count in range(0, 3):
            out[count] = coordinates[count]

        for count in range(3, 6):
            out[count] = velocities[count - 3]

        return out, radius_vector
    else:
        return 'Shit in kepler2descart out_type parameter'


class Body:

    def __init__(self, a_q, e, i, om, w, v_m, param='axis', anomaly='true', dim='deg'):
        """Initiates object and sets a shape of orbit and body's position on it

        :param a_q: semi major axis or perihelion distance
        :param e: eccentricity
        :param i: inclination
        :param om: ascending node
        :param w: argument of perihelion
        :param v_m: true or mean anomaly
        :param param: sets what value is used as the first parameter:
                'axis' - semi major axis (default),
                'distance' - perihelion distance
        :param anomaly: sets what anomaly is used:
                'true' - true anomaly (default)
                'mean' - mean anomaly
        :param dim: sets the dimension of the angular values:
                'deg' - degrees (default),
                'rad' - radians
        :return: None

        """
        self.__eccentricity = e
        if dim == 'deg':
            self.__inclination = math.radians(i)
            self.__ascendingNode = math.radians(om)
            self.__argumentPerihelion = math.radians(w)
            v_m = math.radians(v_m)
        elif dim == 'rad':
            self.__inclination = i
            self.__ascendingNode = om
            self.__argumentPerihelion = w

        if param == 'axis':
            self.__semimajorAxis = a_q
            self.__perihelionDistance = a_q * (1.0 - e)
        elif param == 'distance':
            self.__perihelionDistance = a_q
            self.__semimajorAxis = a_q / (1.0 - e)

        if anomaly == 'true':
            self.__trueAnomaly = v_m
            self.__meanAnomaly = true2mean_anomaly(self.__eccentricity, v_m)
        elif anomaly == 'mean':
            self.__meanAnomaly = v_m
            self.__trueAnomaly = mean2true_anomaly(self.__semimajorAxis, self.__eccentricity, v_m)

        [self.__coordinates, self.__velocities] = kepler2descart([self.__semimajorAxis, self.__eccentricity,
                                                                  self.__inclination, self.__ascendingNode,
                                                                  self.__argumentPerihelion, self.__trueAnomaly],
                                                                 dim='rad')
        self.__time_object = None

    def set_orbit_shape(self, a_q, e, i, om, w, param='axis', dim='deg'):
        """Sets the shape of the orbit

        :param a_q: semi major axis or perihelion distance
        :param e: eccentricity
        :param i: inclination
        :param om: ascending node
        :param w: argument of perihelion
        :param param: sets what value is used as the first parameter:
                'axis' - semi major axis (default),
                'distance' - perihelion distance
        :param dim: sets the dimension of the angular values:
                'deg' - degrees (default),
                'rad' - radians
        :return: None

        """
        self.__eccentricity = e
        if dim == 'deg':
            self.__inclination = math.radians(i)
            self.__ascendingNode = math.radians(om)
            self.__argumentPerihelion = math.radians(w)
        elif dim == 'rad':
            self.__inclination = i
            self.__ascendingNode = om
            self.__argumentPerihelion = w

        if param == 'axis':
            self.__semimajorAxis = a_q
            self.__perihelionDistance = a_q * (1.0 - e)
        elif param == 'distance':
            self.__perihelionDistance = a_q
            self.__semimajorAxis = a_q / (1.0 - e)

    def set_anomaly(self, v_m, anomaly='true', dim='deg'):
        """Sets the position of the body on an orbit by the anomaly

        :param v_m: true or mean anomaly
        :param anomaly: sets what anomaly is used:
                'true' - true anomaly (default)
                'mean' - mean anomaly
        :param dim: sets the dimension of the angular values:
                'deg' - degrees (default),
                'rad' - radians
        :return: None
        """
        if dim == 'deg':
            v_m = math.radians(v_m)

        if anomaly == 'true':
            self.__trueAnomaly = v_m
            self.__meanAnomaly = true2mean_anomaly(self.__eccentricity, v_m)
        elif anomaly == 'mean':
            self.__meanAnomaly = v_m
            self.__trueAnomaly = mean2true_anomaly(self.__semimajorAxis, self.__eccentricity, v_m)

        [self.__coordinates, self.__velocities] = kepler2descart([self.__semimajorAxis, self.__eccentricity,
                                                                  self.__inclination, self.__ascendingNode,
                                                                  self.__argumentPerihelion, self.__trueAnomaly],
                                                                 dim='rad')

    def set_position(self, x, y, z, vx, vy, vz):
        """Sets the position of the body by coordinates and velocities"""
        self.__coordinates = [x, y, z]
        self.__velocities = [vx, vy, vz]
        [a, e, i, om, w, v] = descart2kepler([x, y, z], [vx, vy, vz])
        self.__semimajorAxis = a
        self.__eccentricity = e
        self.__inclination = i
        self.__ascendingNode = om
        self.__argumentPerihelion = w
        self.__perihelionDistance = a * (1.0 - e)
        self.__trueAnomaly = v

    def set_vector_position(self, position6):
        """Sets the position of the body on the coordinates and velocities represented like a six-dimensional vector"""
        self.__coordinates = [position6[0], position6[1], position6[2]]
        self.__velocities = [position6[3], position6[4], position6[5]]
        [a, e, i, om, w, v] = descart2kepler(self.__coordinates, self.__velocities)
        self.__semimajorAxis = a
        self.__eccentricity = e
        self.__inclination = i
        self.__ascendingNode = om
        self.__argumentPerihelion = w
        self.__perihelionDistance = a * (1.0 - e)
        self.__trueAnomaly = v

    def set_epoch(self, julian_date):
        """Sets the epoch for position of the body and save it to the Astropy Time object"""
        self.__time_object = Time([julian_date], format='jd')

    def set_date(self, year, month, day, hours, minutes, seconds):
        """Sets the date for position of the body and save it to the Astropy Time object"""
        if not isinstance(seconds, int):    # the datetime module works with integer
            microseconds = seconds - int(seconds)
            microseconds *= 10 ** 6
            date = datetime.datetime(year, month, day, hours, minutes, int(seconds), int(microseconds))
        else:
            date = datetime.datetime(year, month, day, hours, minutes, seconds)

        self.__time_object = Time([date.isoformat()], format='isot', scale='utc')

    def get_position(self):
        return self.__coordinates, self.__velocities

    def get_orbit(self, param='axis', anomaly='true', dim='deg'):
        """Returns the Keplerian elements of the orbit

        :param param: sets what value is used as the first parameter:
                'axis' - semi major axis (default),
                'distance' - perihelion distance
        :param anomaly: sets what anomaly is used:
                'true' - true anomaly (default)
                'mean' - mean anomaly
        :param dim: sets the dimension of the angular values:
                'deg' - degrees (default),
                'rad' - radians
        :return: list of the Keplerian elements

        """
        orbit = []
        if param == 'axis':
            orbit.append(self.__semimajorAxis)
        elif param == 'distance':
            orbit.append(self.__perihelionDistance)

        orbit.append(self.__eccentricity)
        orbit.append(self.__inclination)
        orbit.append(self.__ascendingNode)
        orbit.append(self.__argumentPerihelion)
        if anomaly == 'true':
            orbit.append(self.__trueAnomaly)
        elif anomaly == 'mean':
            orbit.append(self.__meanAnomaly)

        if dim == 'deg':
            orbit[2::] = np.degrees(orbit[2::])

        return orbit

    def get_time(self):
        """Returns Astropy Time object"""
        return self.__time_object

    def get_date(self, form='iso'):
        dictionary = {'jd': self.__time_object.jd[0],
                      'iso': self.__time_object.iso[0],
                      'isot': self.__time_object.isot[0],
                      'byear': self.__time_object.byear[0]}
        return dictionary[form]

    def get_info(self):
        info = self.get_orbit(anomaly='mean')
        info.append(self.get_date('byear'))
        info.append(self.get_date('jd'))
        return np.array(info)


class Stream:

    def __init__(self, parent, num_of_particles):
        self.__particles = []
        self.__anomaly_of_ej = []
        orbit = parent.get_orbit()
        rc = 1.0
        dens = 1.9
        mass = 7.6E-06
        rpart = math.pow(0.75 * mass / (math.pi * dens), 1.0 / 3.0)  # radius = [cm]
        q0 = orbit[0] * (1.0 - orbit[1])
        p = orbit[0] * (1.0 - orbit[1] ** 2)
        Q = orbit[0] * (1.0 + orbit[1])

        for count in range(0, num_of_particles):
            r = math.pow((1 / (random.random() * (math.pow(Q, -3.0) - math.pow(q0, -3.0))
                               + math.pow(q0, -3.0))), 1.0 / 3.0)
            v0 = (p - r) / (r * orbit[1])
            v0 = math.acos(v0)
            if random.random() < 0.5:
                v0 = 2 * math.pi - v0
            self.__anomaly_of_ej.append(v0)
            parent.set_anomaly(v0)
            [x30, vel30] = parent.get_position()
            cos_T = 1.0 - 2.0 * random.random()
            sin_T = (1.0 - cos_T ** 2) ** 2
            fi = 2.0 * math.pi * random.random()
            c = math.sqrt(1 / (dens * rpart * math.pow(np.dot(x30, x30), 9.0 / 8.0))
                          - 0.013 * rc) * math.sqrt(rc) * 656  # cm/sec
            c *= (6.68458712E-9 / 1.15740741)  # au/day
            c3 = np.array([c * cos_T, c * sin_T * math.cos(fi), c * sin_T * math.sin(fi)])
            mark = descart2kepler(x30, np.array(vel30) + c3)

            particle = Body(mark[0], mark[1], mark[2], mark[3], mark[4], mark[5], dim='rad')
            n = math.sqrt((MU / mark[0] ** 3))
            mean_anomaly = particle.get_orbit(anomaly='mean', dim='rad')[5]
            if mean_anomaly >= math.pi:
                jd = parent.get_date('jd') - mean_anomaly / n
            else:
                jd = parent.get_date('jd') + mean_anomaly / n

            particle.set_epoch(jd)

            self.__particles.append(particle)

    def __getitem__(self, item):
        return self.__particles[item]

    def get_info(self):
        info = np.zeros([len(self.__particles), 10])
        for count in range(0, len(self.__particles)):
            info[count, 0:8] = self.__particles[count].get_info()
            info[count, 8] = count + 1
            info[count, 9] = self.__anomaly_of_ej[count]

        return info
