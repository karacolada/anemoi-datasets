# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""UKV (Met Office Unified Model, regional) GRIB flavour.

Maps the Met Office UM ``shortName`` values found in the UKV "Wholesale"
trajectory GRIB files to CF ``param`` identifiers. The mapping covers the
full Wholesale dataset: near-surface quantities (Wholesale 1), cloud, snow
and surface-flux products (Wholesale 2), and pressure-level dynamics
(Wholesale 3). Only parameters whose CF identifier is well established are
mapped; genuinely ambiguous ones (notably ``h`` and ``hcct``) are left
unmapped so they keep their raw ``shortName`` rather than being mislabelled.

Visibility is carried twice per step in the Wholesale files: as a
deterministic forecast (``productDefinitionTemplateNumber == 0``) and as a
probability-of-low-visibility product
(``productDefinitionTemplateNumber == 5``). Only the deterministic product
is mapped to ``visibility``; the probability product is left unmapped so
that the two cannot collide on the same variable.

The resulting rule list (:data:`UKV_FLAVOUR`) is passed to
:class:`anemoi.transform.flavour.RuleBasedFlavour` via the ``flavour``
argument of the ``grib`` source, e.g.::

    from anemoi.datasets.create.sources.ukv import UKV_FLAVOUR

    source:
      type: grib
      path: "data/{date:strftime(%Y%m%d%H%M)}_u1096_ng_umqv_Wholesale{wholesale}.grib"
      wholesale: [1, 2, 3, 5]
      flavour: <UKV_FLAVOUR>
"""

from typing import Any

# Mapping of Met Office shortName -> CF param identifier for the parameters
# carried in the UKV Wholesale files. Only parameters whose CF identifier is
# well established are mapped; parameters whose CF identifier could not be
# verified against the archive are intentionally left out so that they keep
# their raw shortName as the ``param`` identifier rather than being
# mislabelled. In particular ``h`` and ``hcct`` are genuinely ambiguous
# (different meanings in different UM products) and stay unmapped.
_UKV_PARAM_MAP: dict[str, str] = {
    "t": "temperature",
    "r": "relative_humidity",
    "gh": "geopotential_height",
    "ws": "wind_speed",
    "wdir": "wind_direction",
    "prmsl": "pressure_reduced_to_mean_sea_level",
    "dpt": "dewpoint_temperature",
    "prate": "precipitation_rate",
    "vis": "visibility",
    "10si": "wind_speed_10m",
    "10wdir": "wind_direction_10m",
    "cbh": "cloud_base_height",
    "hcc": "high_cloud_area_fraction",
    "lcc": "low_cloud_area_fraction",
    "mcc": "medium_cloud_area_fraction",
    "sde": "snow_depth",
    "sdlwrf": "surface_downwelling_longwave_flux",
    "sdswrf": "surface_downwelling_shortwave_flux",
}

#: Rule list for :class:`RuleBasedFlavour`. Each rule matches a Met Office
#: ``shortName`` and sets the CF ``param`` identifier. The ``vis`` rule is
#: special-cased: the Wholesale files carry two ``vis`` messages per step,
#: the deterministic visibility forecast
#: (``productDefinitionTemplateNumber == 0``) and a probability-of-low-
#: visibility product (``productDefinitionTemplateNumber == 5``). Only the
#: deterministic product is mapped to ``visibility`` so that the two cannot
#: collide on the same variable.
UKV_FLAVOUR: list[dict[str, Any]] = [
    {
        "match": {"shortName": "vis", "productDefinitionTemplateNumber": 0},
        "result": {"param": "visibility"},
    },
    *[
        {"match": {"shortName": short_name}, "result": {"param": param}}
        for short_name, param in _UKV_PARAM_MAP.items()
        if short_name != "vis"
    ],
]
