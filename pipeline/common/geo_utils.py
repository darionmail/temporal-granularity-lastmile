"""
Zajednicke geografske utility funkcije.
Projekcija lat/lon -> lokalni metricki sustav (x, y u metrima), prema formulaciji iz rada:

x = (lambda - lambda0) * Mlon
y = (phi - phi0) * Mlat

gdje je Mlat ~= 111320 m/deg, Mlon ~= 111320 * cos(phi0) m/deg
(phi0, lambda0) = centar studijske regije
"""
import numpy as np

# Centar Zagreba (referentna tocka za projekciju, kao u radu)
ZAGREB_LAT = 45.8150  # phi0
ZAGREB_LON = 15.9819  # lambda0

M_LAT = 111320.0  # m/deg
M_LON = 111320.0 * np.cos(np.radians(ZAGREB_LAT))  # m/deg, korigirano za sirinu


def latlon_to_xy(lat, lon):
    """
    Konverzija geografskih koordinata (lat=phi, lon=lambda) u lokalne metricke (x, y) u metrima.
    Prima skalare ili numpy/pandas array-like.
    """
    x = (lon - ZAGREB_LON) * M_LON
    y = (lat - ZAGREB_LAT) * M_LAT
    return x, y


def xy_to_latlon(x, y):
    """Inverzna konverzija, metricke (x,y) -> geografske (lat, lon). Korisno za vizualizaciju na karti."""
    lon = x / M_LON + ZAGREB_LON
    lat = y / M_LAT + ZAGREB_LAT
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distanca u km izmedju dvije tocke (ili nizova tocaka)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def euclidean_m(x1, y1, x2, y2):
    """Euklidska distanca u metrima izmedju tocaka u lokalnom metrickom sustavu."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
