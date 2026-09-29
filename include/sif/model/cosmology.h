/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file cosmology.h
 * @brief The background cosmology: an expansion history, and the comoving
 * distances it implies.
 *
 * What turns a survey's redshifts into positions (see
 * sif_field_convert_sky_coordinates()). The dark energy is the
 * Chevallier-Polarski-Linder one, w(a) = w0 + wa (1 - a), so w0waCDM, wCDM
 * and LambdaCDM are the same struct with different numbers:
 *
 *   E(z)^2 = omega_r (1+z)^4 + omega_m (1+z)^3 + omega_k (1+z)^2
 *          + omega_de (1+z)^(3 (1 + w0 + wa)) exp(-3 wa z / (1+z)),
 *
 * with omega_k = 1 - omega_m - omega_de - omega_r: the curvature is whatever
 * the other densities leave, so a flat model is one whose densities sum to 1.
 *
 * **Distances are in Mpc/h.** The Hubble distance c / H0 is 2997.92458 Mpc/h
 * whatever H0 is, so no h is needed, and the positions come out in the units
 * the GADGET reader gives a simulation. A survey analysed in Mpc has to be
 * scaled by 1/h afterwards.
 */

#ifndef SIF_MODEL_COSMOLOGY_H
#define SIF_MODEL_COSMOLOGY_H

#include "sif/core/macros.h"

/**
 * @brief What a set of positions is: a field's tracers (sif_field_t::units)
 * or a catalogue's void centres (sif_catalogue_t::units).
 *
 * The two systems a cosmology converts between, and the reason it does:
 * sif_field_convert_sky_coordinates() takes a survey from the sky into the
 * Cartesian frame the finders work in, and sif_catalogue_to_sky() takes the
 * voids found back out.
 */
typedef enum {
  /**
   * Cartesian coordinates in x, y and z, in the units of the box -- comoving
   * Mpc/h, observer at the origin, for a survey. What every function that
   * bins, sorts, bounds or moves positions needs, and what every reader in
   * sif produces unless told otherwise.
   */
  SIF_COORDINATES_CARTESIAN = 0,
  /**
   * Sky coordinates: x holds the right ascension and y the declination, both
   * in degrees, and z the redshift.
   */
  SIF_COORDINATES_SKY
} sif_coordinates_t;

/** @brief c / (100 km/s/Mpc): the Hubble distance, in Mpc/h. */
#define SIF_HUBBLE_DISTANCE 2997.92458

/**
 * @brief The parameters of a w0waCDM background.
 *
 * Densities are today's, in units of the critical density. Curvature is not a
 * field of its own: it is 1 minus the sum of the three.
 */
typedef struct {
  /** Matter: cold dark matter and baryons. */
  double omega_m;
  /** Dark energy. */
  double omega_de;
  /** Radiation, massless neutrinos included; 0 is fine below z of a few
   * hundred, where it no longer matters. */
  double omega_r;
  /** Dark-energy equation of state today; -1 for a cosmological constant. */
  double w0;
  /** Its evolution, w(a) = w0 + wa (1 - a); 0 for a constant w. */
  double wa;
} sif_cosmology_t;

/**
 * @brief A flat LambdaCDM cosmology with matter density @p om.
 *
 * @code
 * sif_field_convert_sky_coordinates(field, SIF_COSMOLOGY_FLAT_LCDM(0.31));
 * @endcode
 *
 * @warning A compound literal, whose lifetime ends with the enclosing block:
 * pass it straight to a function, do not store the pointer.
 */
#define SIF_COSMOLOGY_FLAT_LCDM(om)                                            \
  (&(sif_cosmology_t){.omega_m = (om),                                         \
    .omega_de = 1.0 - (om),                                                    \
    .omega_r = 0.0,                                                            \
    .w0 = -1.0,                                                                \
    .wa = 0.0})

/**
 * @brief Line-of-sight comoving distance to redshift @p z.
 *
 * D_C(z) = (c / H0) * integral from 0 to z of dz' / E(z'), integrated to
 * about machine precision. This is the distance a survey's tracers are placed
 * at along their line of sight; in a curved model it is not the transverse
 * comoving distance, which angular sizes would need.
 *
 * @param cosmo The cosmology.
 * @param z Redshift, at least 0.
 * @param[out] distance D_C(z), in Mpc/h.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument, a negative or
 * non-finite redshift, or parameters that are not finite or give a negative
 * density; SIF_ERR_RANGE if E(z)^2 is not positive somewhere on [0, z] -- a
 * model with no expansion history out to that redshift (a bounce, say).
 */
SIF_NODISCARD int sif_cosmology_comoving_distance(
  const sif_cosmology_t* cosmo, double z, double* distance);

#endif /* SIF_MODEL_COSMOLOGY_H */
