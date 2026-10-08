# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import glob
import logging
import os
from typing import Any

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.flavour import RuleBasedFlavour
from anemoi.transform.grids import grid_registry
from earthkit.data import from_source
from earthkit.data.utils.dates import to_timedelta
from earthkit.data.utils.patterns import Pattern

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)


def check(ds: Any, paths: list[str], **kwargs: Any) -> None:
    """Check if the dataset matches the expected number of fields.

    Parameters
    ----------
    ds : Any
        The dataset to check.
    paths : list of str
        List of paths to the GRIB files.
    **kwargs : Any
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If the number of fields does not match the expected count.
    """
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    # in the case of static data (e.g repeated dates) dates might be empty
    if len(ds) != count and kwargs.get("dates", []) == []:
        LOG.warning(
            f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, paths={paths})"
            f" Received empty dates - assuming this is static data."
        )
        return

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, paths={paths})")


def _expand(paths: list[str]) -> Any:
    """Expand the given paths using glob.

    Parameters
    ----------
    paths : list of str
        List of paths to expand.

    Returns
    -------
    Any
        The expanded paths.
    """
    for path in paths:
        cnt = 0
        for p in glob.glob(path):
            yield p
            cnt += 1
        if cnt == 0:
            yield path


# Keys that must never be forwarded to ``.sel()`` as field filters in the
# forecast path. ``valid_datetime`` is a validity-date concept and is not a
# meaningful filter when files are indexed by (basetime, step).
_NEVER_SEL_KEYS = frozenset({"valid_datetime"})


def _step_hours(step: Any) -> int:
    """Normalise a GRIB step value to a whole number of hours.

    GRIB files may report the step as an ``int`` (hours), a ``str`` (e.g.
    ``"0m"`` for the analysis, ``"36"`` or ``"36h"``), or a
    ``datetime.timedelta``. The trajectory pipeline performs
    ``int(field.metadata("step"))`` so a non-int step (notably the string
    ``"0m"``) would crash it; this helper converts every representation to
    an ``int`` number of hours.

    Parameters
    ----------
    step : Any
        The step value as reported by the field metadata.

    Returns
    -------
    int
        The step expressed in whole hours.

    Raises
    ------
    ValueError
        If the step is not a whole number of hours.
    """
    td = to_timedelta(step)
    seconds = int(td.total_seconds())
    if seconds % 3600 != 0:
        raise ValueError(f"Expected a whole-hour step, got {step!r} ({td})")
    return seconds // 3600


@source_registry.register("grib")
class GribSource(Source):

    def __init__(
        self,
        context: Any,
        path: str | list[str],
        flavour: str | list[Any] | None = None,
        grid_definition: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise the GRIB source.

        Parameters
        ----------
        context : Any
            The context in which the source is created.
        path : str or list of str
            Path or list of paths to the GRIB files.
        flavour : str or list of rule, optional
            Flavour rules (a path to a rules file or a list of rule
            definitions), by default None.
        grid_definition : dict of str to Any, optional
            Grid definition configuration to create a Grid object, by default None.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments forwarded to ``.sel()``.
        """
        super().__init__(context)
        self.path = path
        self.flavour = RuleBasedFlavour(flavour) if flavour is not None else None
        self.grid = grid_registry.from_config(grid_definition) if grid_definition is not None else None
        self.args = args
        self.kwargs = kwargs

    def execute_valid_dates(self, dates: ValidDates) -> ekd.FieldList:
        """Load data from the GRIB files for the given dates.

        Parameters
        ----------
        dates : ValidDates
            The validity-time argument from the pipeline.

        Returns
        -------
        ekd.FieldList
            The loaded dataset.
        """
        given_paths = self.path if isinstance(self.path, list) else [self.path]

        ds = from_source("empty")
        dates = [d.isoformat() for d in dates]

        for path in given_paths:

            # do not substitute if not needed
            if "{" not in path:
                paths = [path]
            else:
                paths = Pattern(path).substitute(*self.args, date=dates, allow_extra=True, **self.kwargs)

            for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
                if name in self.kwargs:
                    raise ValueError(f"MARS interpolation parameter '{name}' not supported")

            for path in _expand(paths):
                self.context.trace("📁", "PATH", path)

                if isinstance(path, str) and (path.startswith("ec:") or path.startswith("ectmp:")):
                    from anemoi.datasets.create.ecfs import get_ecfs_file

                    path = get_ecfs_file(path)

                s = from_source("file", path)
                if self.flavour is not None:
                    s = self.flavour.map(s)
                sel_kwargs = self.kwargs.copy()
                if dates != []:
                    sel_kwargs["valid_datetime"] = dates
                s = s.sel(**sel_kwargs)
                ds = ds + s

        # if kwargs and not context.partial_ok:
        # BACK    check(ds, given_paths, valid_datetime=dates, **kwargs)

        if self.grid is not None:
            ds = new_fieldlist_from_list([new_field_from_grid(f, self.grid) for f in ds])

        if len(ds) == 0:
            LOG.warning(f"No fields found for {dates} in {given_paths} (kwargs={self.kwargs})")

        return ds

    def _path_params(self) -> set[str]:
        """Return the set of variable names used in the path templates.

        These are the keyword arguments that drive file selection (e.g.
        ``wholesale``) and must not be forwarded to ``.sel()`` as field
        filters, because a field never carries such a metadata key and the
        selection would silently match nothing.

        Returns
        -------
        set of str
            The variable names appearing in the path templates.
        """
        given_paths = self.path if isinstance(self.path, list) else [self.path]
        params: set[str] = set()
        for template in given_paths:
            if "{" in template:
                params.update(Pattern(template).names)
        return params

    def _requested_keys(
        self, dates: ForecastDates
    ) -> tuple[set[tuple[str, int, int]], list[datetime.datetime]]:
        """Derive the requested (date, time, step) keys and distinct basetimes.

        Parameters
        ----------
        dates : ForecastDates
            The forecast-date argument.

        Returns
        -------
        tuple
            ``(keys, basetimes)`` where ``keys`` is the set of
            ``(date_str, time_int, step_hours_int)`` triples that were
            requested and ``basetimes`` is the ordered list of distinct
            basetimes.
        """
        keys: set[tuple[str, int, int]] = set()
        basetimes: list[datetime.datetime] = []
        seen: set[datetime.datetime] = set()
        for valid_time, basetime in dates.items:
            keys.add(
                (
                    basetime.strftime("%Y%m%d"),
                    int(basetime.strftime("%H%M")),
                    _step_hours(valid_time - basetime),
                )
            )
            if basetime not in seen:
                seen.add(basetime)
                basetimes.append(basetime)
        return keys, basetimes

    def _resolve_paths(self, basetimes: list[datetime.datetime]) -> list[str]:
        """Resolve the configured path templates to concrete file paths.

        Each template is substituted **per basetime** with single (non-list)
        ``date``/``time`` values so that ``Pattern.substitute`` does not fan
        out a Cartesian product over basetimes. List-valued keyword arguments
        (e.g. ``wholesale=[1, 2, 3, 5]``) still expand to one path per value.

        Parameters
        ----------
        basetimes : list of datetime
            The distinct basetimes to resolve paths for.

        Returns
        -------
        list of str
            The resolved, de-duplicated file paths.
        """
        given_paths = self.path if isinstance(self.path, list) else [self.path]
        resolved: list[str] = []
        seen: set[str] = set()
        path_params = self._path_params()
        sub_kwargs = {
            k: v for k, v in self.kwargs.items() if k in path_params and k not in ("date", "time")
        }

        def _add(out: Any) -> None:
            as_list = [out] if isinstance(out, str) else list(out)
            for p in _expand(as_list):
                if p not in seen:
                    seen.add(p)
                    resolved.append(p)

        for template in given_paths:
            if "{" in template:
                for basetime in basetimes:
                    out = Pattern(template).substitute(
                        *self.args, date=basetime, time=basetime, allow_extra=True, **sub_kwargs
                    )
                    _add(out)
            else:
                _add(template)

        return resolved

    def execute_forecast_dates(self, dates: ForecastDates) -> ekd.FieldList:
        """Load data from the GRIB files for the given forecast requests.

        Unlike :meth:`execute_valid_dates`, the files are indexed by
        *basetime* and *step* rather than by validity time. Each configured
        path template is resolved per basetime, the resulting files are
        loaded, and only the fields whose ``(basetime, step)`` matches a
        requested key are returned.

        The step of every returned field is normalised to an integer number
        of hours via :func:`new_field_with_metadata`, because the trajectory
        pipeline performs ``int(field.metadata("step"))`` and UKV GRIB files
        report the analysis step as the string ``"0m"``.

        Parameters
        ----------
        dates : ForecastDates
            The forecast-date argument from the pipeline.

        Returns
        -------
        ekd.FieldList
            The loaded dataset.
        """
        keys, basetimes = self._requested_keys(dates)

        for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
            if name in self.kwargs:
                raise ValueError(f"MARS interpolation parameter '{name}' not supported")

        path_params = self._path_params()
        sel_kwargs = {
            k: v for k, v in self.kwargs.items() if k not in path_params and k not in _NEVER_SEL_KEYS
        }

        fields: list[Any] = []
        for path in self._resolve_paths(basetimes):
            self.context.trace("📁", "PATH", path)

            if isinstance(path, str) and (path.startswith("ec:") or path.startswith("ectmp:")):
                from anemoi.datasets.create.ecfs import get_ecfs_file

                path = get_ecfs_file(path)

            if not os.path.exists(path):
                raise FileNotFoundError(f"GRIB file not found: {path}")

            s = from_source("file", path)
            if self.flavour is not None:
                s = self.flavour.map(s)
            if sel_kwargs:
                s = s.sel(**sel_kwargs)

            for f in s:
                key = (
                    str(int(f.metadata("date"))),
                    int(f.metadata("time") or 0),
                    _step_hours(f.metadata("step")),
                )
                if key in keys:
                    fields.append(new_field_with_metadata(f, step=key[2]))

        if self.grid is not None:
            fields = [new_field_from_grid(f, self.grid) for f in fields]

        ds = new_fieldlist_from_list(fields)

        if len(ds) == 0:
            LOG.warning(
                f"No fields found for {len(keys)} requested (date, time, step) keys in "
                f"{self.path} (kwargs={self.kwargs})"
            )

        return ds
