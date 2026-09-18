from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy.io import fits

from sunpy.coordinates import sun
from astropy.coordinates import get_sun, get_body


# ============================================================
# YOUR FILE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FITS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hdr"
    / "solar_corona_hdr_linear.fits"
)


# ============================================================
# READ HEADER
# ============================================================

with fits.open(FITS_PATH) as hdul:
    h = hdul[0].header


moon_x = float(h["MOON-X"])
moon_y = float(h["MOON-Y"])

cx = 1551.5
cy = 1145.5

latitude = float(h["OBSLAT"])
longitude = float(h["OBSLONG"])


print("=" * 70)
print("DEB 2026 SOLAR PA ORIENTATION")
print("=" * 70)

print(f"FITS: {FITS_PATH}")
print(f"Moon image position: ({moon_x:.6f}, {moon_y:.6f}) px")
print(f"Solar center:        ({cx:.6f}, {cy:.6f}) px")
print(f"Observer:            {latitude:.8f}, {longitude:.8f}")


# ============================================================
# OBSERVATION TIME
# ============================================================
#
# OBJECT = partial_2026-08-12_182629
#
# The eclipse maximum near Valladolid occurred around
# 18:30 UTC, so this timestamp is interpreted as UTC.
#
# ============================================================

time = Time(
    "2026-08-12T18:26:29",
    scale="utc"
)

print(f"Observation time: {time.isot}")


# ============================================================
# OBSERVER
# ============================================================
location = EarthLocation.from_geodetic(
    lon=longitude * u.deg,
    lat=latitude * u.deg,
    height=0 * u.m,
)


# ============================================================
# IMAGE MOON VECTOR
# ============================================================

dx = moon_x - cx

dy = moon_y - cy

print()
print("IMAGE MOON VECTOR")
print("-----------------")
print(f"dX = {dx:+.6f} px")
print(f"dY = {dy:+.6f} px")


# Image coordinate angle:
#
# +X = right
# +Y = down
#
# Therefore atan2(dY, dX) gives the visual/image angle
# measured clockwise from +X.

image_angle = (
    np.degrees(
        np.arctan2(
            dy,
            dx
        )
    )
    % 360.0
)

print(
    f"Moon image angle = "
    f"{image_angle:.6f}°"
)


# ============================================================
# SOLAR P ANGLE
# ============================================================

P = sun.P(time)

P_deg = P.to_value(
    u.deg
)

print()
print("SOLAR ORIENTATION")
print("-----------------")

print(
    f"Solar P angle = "
    f"{P_deg:.6f}°"
)


# ============================================================
# SUN AND MOON SKY POSITIONS
# ============================================================

sun_coord = get_sun(
    time
)

moon_coord = get_body(
    "moon",
    time,
    location
)


# Transform both to the same apparent topocentric frame.
frame = AltAz(
    obstime=time,
    location=location
)

sun_altaz = sun_coord.transform_to(
    frame
)

moon_altaz = moon_coord.transform_to(
    frame
)

print()
print("SKY POSITIONS")
print("-------------")

print(
    f"Sun altitude  = "
    f"{sun_altaz.alt.to_value(u.deg):.6f}°"
)

print(
    f"Sun azimuth   = "
    f"{sun_altaz.az.to_value(u.deg):.6f}°"
)

print(
    f"Moon altitude = "
    f"{moon_altaz.alt.to_value(u.deg):.6f}°"
)

print(
    f"Moon azimuth  = "
    f"{moon_altaz.az.to_value(u.deg):.6f}°"
)


# ============================================================
# MOON RELATIVE TO SUN
# ============================================================
#
# Calculate the apparent position angle of the Moon relative
# to the Sun in the local sky.
#
# PA here is measured eastward from celestial north.
# ============================================================

moon_sun_pa_celestial = (
    sun_coord.position_angle(
        moon_coord
    ).to_value(
        u.deg
    )
    % 360.0
)

print()
print("CELESTIAL MOON-SUN GEOMETRY")
print("----------------------------")

print(
    f"Moon PA from Sun "
    f"(celestial north) = "
    f"{moon_sun_pa_celestial:.6f}°"
)


# ============================================================
# CONVERT CELESTIAL PA TO SOLAR PA
# ============================================================
#
# Solar P is the rotation of solar north relative to
# celestial north.
#
# Therefore:
#
# PA_sun = PA_celestial - P
#
# modulo 360°.
# ============================================================

moon_sun_pa_solar = (
    moon_sun_pa_celestial
    - P_deg
) % 360.0

print(
    f"Moon PA from Sun "
    f"(solar north) = "
    f"{moon_sun_pa_solar:.6f}°"
)


# ============================================================
# CAMERA ROTATION
# ============================================================
#
# The image angle above is measured clockwise from image +X.
#
# The solar PA is measured counter-clockwise/eastward from
# solar north.
#
# We therefore need the handedness convention to be explicit.
#
# First calculate the simple rotation assuming the image has
# not been mirrored.
# ============================================================

camera_offset = (
    moon_sun_pa_solar
    - image_angle
) % 360.0

print()
print("CAMERA ORIENTATION")
print("------------------")

print(
    f"Candidate image → solar PA offset = "
    f"{camera_offset:.6f}°"
)


# Also provide the opposite-hand solution. This is important
# because one Moon position alone cannot distinguish a rotation
# from a reflection.

mirror_offset = (
    moon_sun_pa_solar
    + image_angle
) % 360.0

print(
    f"Candidate reflected-image offset = "
    f"{mirror_offset:.6f}°"
)


print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "The two candidate offsets differ because one image vector "
    "cannot by itself distinguish camera rotation from a mirror flip."
)

print(
    "We should use the known Moon motion or another orientation "
    "reference to choose between them."
)

print("=" * 70)