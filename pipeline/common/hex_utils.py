"""
Heksagonalna geometrija: axial koordinate, snap-cover-cap konstrukcija,
povrsina/inradius, prema formulaciji iz rada (Sekcija 3 - Spatial Discretization
and Hexagonal Geometry).

Pointy-topped regularni heksagon sa stranicom s (metri):
  A(s) = (3*sqrt(3)/2) * s^2          [povrsina]
  r(s) = (sqrt(3)/2) * s              [inradius - okomita distanca centar->rub]

Axial -> Cartesian (pointy-topped):
  x = s*sqrt(3) * (q + r/2)
  y = (3/2) * s * r

Cartesian -> axial (inverz):
  q = (sqrt(3)/3)*(x/s) - (1/3)*(y/s)
  r = (2/3)*(y/s)
"""
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate


SQRT3 = np.sqrt(3.0)


def hex_area(s):
    """Povrsina pointy-topped regularnog heksagona sa stranicom s (metri)."""
    return (3 * SQRT3 / 2.0) * s ** 2


def hex_inradius(s):
    """Inradius (centar -> rub, okomito) za stranicu s."""
    return (SQRT3 / 2.0) * s


def axial_to_cartesian(q, r, s):
    x = s * SQRT3 * (q + r / 2.0)
    y = 1.5 * s * r
    return x, y


def cartesian_to_axial(x, y, s):
    q = (SQRT3 / 3.0) * (x / s) - (1.0 / 3.0) * (y / s)
    r = (2.0 / 3.0) * (y / s)
    return q, r


def axial_round(q, r):
    """
    Cube-coordinate rounding za axial koordinate (Patel, 2013).
    Osigurava da snapping na lattice bude konzistentan (q+r+s_cube == 0 invarijanta).
    """
    x_cube = q
    z_cube = r
    y_cube = -x_cube - z_cube

    rx = round(x_cube)
    ry = round(y_cube)
    rz = round(z_cube)

    x_diff = abs(rx - x_cube)
    y_diff = abs(ry - y_cube)
    z_diff = abs(rz - z_cube)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return rx, rz  # (q, r)


def snap_to_lattice(x, y, lattice_s):
    """
    Snapa tocku (x, y) u metrima na najblizi centar heksagonalne lattice sa stranicom lattice_s.
    Vraca snapped (x*, y*).
    """
    q, r = cartesian_to_axial(x, y, lattice_s)
    q_round, r_round = axial_round(q, r)
    x_snap, y_snap = axial_to_cartesian(q_round, r_round, lattice_s)
    return x_snap, y_snap


def hex_polygon(center_x, center_y, s):
    """
    Vraca shapely Polygon za pointy-topped regularni heksagon sa zadanim centrom i stranicom s.
    Pointy-topped: vrhovi na uglovima 90, 150, 210, 270, 330, 30 stupnjeva (vrh gore/dolje).
    """
    angles_deg = [90, 150, 210, 270, 330, 30]
    points = []
    for ang in angles_deg:
        rad = np.radians(ang)
        px = center_x + s * np.cos(rad)
        py = center_y + s * np.sin(rad)
        points.append((px, py))
    return Polygon(points)


def snap_cover_cap(center_x, center_y, points_xy, lattice_s=1000.0, s_base=1000.0, s_max=2000.0, s_floor_frac=0.75):
    """
    Snap-Cover-Cap konstrukcija heksagonalnog teritorija (Sekcija 3.4 rada).

    Step 1 (Snap): centar se snapa na najblizu 1km lattice pozciju.
    Step 2 (Cover): minimalna stranica koja pokriva sve dodijeljene tocke:
        s_needed = (2/sqrt(3)) * max_x ||x - c*||
    Step 3 (Cap): s = min(max(s_needed, 0.75*s_base), s_max)

    Parameters:
        center_x, center_y: kandidat centar (metri, prije snapanja)
        points_xy: lista/array (x,y) parova - sve tocke dodijeljene ovom kuriru
        lattice_s: velicina lattice celije za snapping (default 1km kao u radu)
        s_base: base side length for floor calculation (default 1000m)
        s_max: strict upper bound on hexagon side length (default 2000m)
        s_floor_frac: udio s_base za floor (default 0.75)

    Vraca: (x_snap, y_snap, s, polygon)
    """
    x_snap, y_snap = snap_to_lattice(center_x, center_y, lattice_s)

    if len(points_xy) == 0:
        s_needed = s_floor_frac * s_base
    else:
        points_xy = np.asarray(points_xy)
        dists = np.sqrt((points_xy[:, 0] - x_snap) ** 2 + (points_xy[:, 1] - y_snap) ** 2)
        max_dist = dists.max()
        s_needed = (2.0 / SQRT3) * max_dist

    s = min(max(s_needed, s_floor_frac * s_base), s_max)

    poly = hex_polygon(x_snap, y_snap, s)
    return x_snap, y_snap, s, poly


def get_neighbor_centers(center_x, center_y, lattice_s):
    """
    Vraca centar + 6 susjeda na lattice sa stranicom lattice_s (za lokalnu optimizaciju, Stage 5).
    Koristi se za 0.5km refinement lattice u radu.
    """
    q, r = cartesian_to_axial(center_x, center_y, lattice_s)
    q0, r0 = axial_round(q, r)

    candidates = [(q0, r0)]  # trenutna pozicija
    # 6 axial susjeda
    axial_directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for dq, dr in axial_directions:
        candidates.append((q0 + dq, r0 + dr))

    centers = []
    for q_c, r_c in candidates:
        cx, cy = axial_to_cartesian(q_c, r_c, lattice_s)
        centers.append((cx, cy))

    return centers
