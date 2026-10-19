from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import colormaps
from matplotlib.colors import to_rgba

from napari_harpy.core.class_palette import default_categorical_colors, normalize_color_sequence

if TYPE_CHECKING:
    from anndata import AnnData

StyledPaletteSource = Literal["stored", "default_missing", "default_invalid"]
STYLED_PALETTE_SOURCES: tuple[StyledPaletteSource, ...] = ("stored", "default_missing", "default_invalid")

MISSING_CATEGORICAL_COLOR = "#80808099"
MISSING_CONTINUOUS_COLOR = "#80808099"
OVERLAY_CONTINUOUS_COLORMAP = "viridis"
# Curated colormap choices offered in the continuous-coloring dropdown. All are standard matplotlib
# colormaps; the perceptually-uniform sequential maps lead, with viridis (the default) first.
CONTINUOUS_COLORMAP_CHOICES: tuple[str, ...] = (
    "viridis",
    "inferno",
    "magma",
    "plasma",
    "cividis",
    "Reds",
    "Blues",
    "Greens",
    "Purples",
    "Oranges",
    "hot",
    "coolwarm",
    "turbo",
    "gray",
)
STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT = 20
STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR = 100
MAX_UMAP_NEIGHBOR_ENTRIES = 20_000_000


def resolve_continuous_colormap(name: str | None) -> str:
    """Return a valid matplotlib colormap name, falling back to the viridis default.

    Guards both ``None`` (unset) and stale/unknown names -- e.g. an old preset referencing a
    colormap no longer registered in this matplotlib -- so continuous styling never raises.
    """
    if isinstance(name, str):
        try:
            colormaps[name]
        except (KeyError, ValueError):
            pass
        else:
            return name
    return OVERLAY_CONTINUOUS_COLORMAP


def validate_styled_palette_source(source: str) -> StyledPaletteSource:
    """Return a validated styled palette source."""
    if source not in STYLED_PALETTE_SOURCES:
        allowed = ", ".join(repr(allowed_source) for allowed_source in STYLED_PALETTE_SOURCES)
        raise ValueError(f"Invalid styled palette source {source!r}. Expected one of: {allowed}.")

    return cast(StyledPaletteSource, source)


def build_string_categorical_values(
    *,
    full_values: pd.Series,
    row_values: pd.Series,
    column_name: str,
) -> tuple[pd.Series, list[object]]:
    """Return viewer-only categorical values for a plain string/object column.

    Parameters
    ----------
    full_values
        Complete source column used for category discovery and cardinality
        warnings. Deriving categories from this complete series keeps the
        temporary categorical palette stable even when the currently rendered
        rows contain only a subset of the source values.
    row_values
        Already-aligned subset or repetition used by the viewer layer, for
        example linked labels table rows for one region or rendered shape rows
        after a ``MultiPolygon`` expands into multiple napari rows.
    column_name
        Source column name used in viewer-facing warning messages.
    """
    normalized_full_values = pd.Series(
        [normalize_string_value(value) for value in full_values],
        index=full_values.index,
        name=column_name,
        dtype="object",
    )
    normalized_row_values = pd.Series(
        [normalize_string_value(value) for value in row_values],
        index=row_values.index,
        name=column_name,
        dtype="object",
    )

    non_missing_values = normalized_full_values.dropna().tolist()
    if has_high_cardinality_string_values(non_missing_values, row_count=len(full_values)):
        unique_count = len({str(value) for value in non_missing_values})
        threshold = string_categorical_warning_threshold(len(full_values))
        logger.warning(
            f"Column `{column_name}` has {unique_count} unique string values across "
            f"{len(full_values)} rows, which exceeds the categorical viewer-coloring threshold of {threshold}. "
            "Harpy will render it with the default categorical palette anyway; "
            "convert the column to pandas categorical dtype to mark this as intentional."
        )
    else:
        logger.warning(
            f"Coercing plain string/object column `{column_name}` to temporary categorical values for viewer coloring."
        )

    categories = list(pd.unique(normalized_full_values.dropna()))
    return normalized_row_values, categories


def string_categorical_warning_threshold(row_count: int) -> int:
    return max(STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT, row_count // STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR)


def has_high_cardinality_string_values(values: Sequence[Any], *, row_count: int) -> bool:
    return len({str(value) for value in values}) > string_categorical_warning_threshold(row_count)


def is_string_like_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    return all(_is_string_scalar(value) for value in non_null.tolist())


def normalize_string_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return str(value)


def normalize_category_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def default_categorical_palette_for_categories(categories: Sequence[object]) -> list[str]:
    return default_categorical_colors(len(categories))


def resolve_table_categorical_palette(
    *,
    table: AnnData,
    column_name: str,
    categories: Sequence[object],
) -> tuple[StyledPaletteSource, list[str]]:
    """Resolve a categorical palette stored in ``table.uns`` for viewer coloring."""
    colors_key = f"{column_name}_colors"
    stored_colors = normalize_color_sequence(table.uns.get(colors_key))
    if stored_colors is None:
        logger.info(
            f"No stored `{colors_key}` palette found in `table.uns`; using the default categorical palette for viewer coloring."
        )
        return "default_missing", default_categorical_palette_for_categories(categories)

    if len(stored_colors) != len(categories):
        logger.warning(
            f"Stored `{colors_key}` palette has {len(stored_colors)} colors for {len(categories)} categories; "
            "using the default categorical palette."
        )
        return "default_invalid", default_categorical_palette_for_categories(categories)

    if not all(is_valid_color(color) for color in stored_colors):
        logger.warning(
            f"Stored `{colors_key}` palette contains invalid color values; using the default categorical palette."
        )
        return "default_invalid", default_categorical_palette_for_categories(categories)

    logger.info(f"Using stored `{colors_key}` palette from `table.uns` for viewer coloring.")
    return "stored", list(stored_colors)


def resolve_expression_matrix(table: AnnData, layer: str | None) -> Any:
    """Return the obs-by-var expression matrix to colour a var source from.

    ``None`` or ``"X"`` selects ``table.X``; any other value selects
    ``table.layers[layer]``. This lets the viewer colour by a gene from an alternative
    expression layer (e.g. ``log1p``, ``raw``, ``smoothed``) instead of ``X`` only.
    """
    if layer is None or layer == "X":
        return table.X
    if layer not in table.layers:
        raise ValueError(f"Expression layer `{layer}` is not available in the selected table.")
    return table.layers[layer]


def var_expression_values(
    table: AnnData,
    *,
    var_name: str,
    obs_positions: np.ndarray,
    layer: str | None = None,
    smoother_key: str | None = None,
) -> np.ndarray:
    """Return a 1-D array of one var's expression at ``obs_positions`` (native matrix dtype).

    Reads from ``table.X`` or ``table.layers[layer]`` (see :func:`resolve_expression_matrix`),
    densifying a sparse column when needed. Raises if any ``obs_positions`` entry is negative
    (an unaligned row), so a careless caller cannot silently wrap to a wrong cell's value.

    When ``smoother_key`` is given, the gene's full column is averaged over each cell's neighbourhood
    live via the row-stochastic operator ``table.obsp[smoother_key]`` (``A @ column``) before the
    ``obs_positions`` rows are taken, so smoothing is computed on demand one gene at a time.
    """
    if var_name not in table.var_names:
        raise ValueError(f"Var `{var_name}` is not available in the selected table.")
    var_index = table.var_names.get_loc(var_name)
    if not isinstance(var_index, (int, np.integer)):
        raise ValueError(f"Var `{var_name}` does not resolve to one unique column in `table.var_names`.")
    obs_positions = np.asarray(obs_positions)
    if obs_positions.size and np.any(obs_positions < 0):
        raise ValueError(f"Could not align linked table rows to `{var_name}` expression for viewer coloring.")
    matrix = resolve_expression_matrix(table, layer)

    if smoother_key is not None:
        if smoother_key not in table.obsp:
            raise ValueError(f"Smoother `{smoother_key}` is not available in `table.obsp`.")
        smoother = table.obsp[smoother_key]
        if getattr(smoother, "shape", None) != (table.n_obs, table.n_obs):
            raise ValueError(
                f"Smoother `{smoother_key}` must be a square ({table.n_obs}, {table.n_obs}) cell-graph operator, "
                f"but has shape {getattr(smoother, 'shape', None)}."
            )
        full_column = matrix[:, int(var_index)]
        full_column = (
            full_column.toarray().reshape(-1)
            if hasattr(full_column, "toarray")
            else np.asarray(full_column).reshape(-1)
        )
        smoothed = np.asarray(smoother @ full_column).reshape(-1)
        return smoothed[obs_positions]

    column_values = matrix[obs_positions, int(var_index)]
    if hasattr(column_values, "toarray"):
        return column_values.toarray().reshape(-1)
    return np.asarray(column_values).reshape(-1)


def _umap_neighbor_operator(table: AnnData, n_neighbors: int) -> tuple[Any, np.ndarray]:
    """Build uniform kNN weights using only finite rows of ``obsm['X_umap']``.

    Include self, matching the precomputed ``smoothed_k100`` workflow. Rows
    without a UMAP coordinate have no edges and must receive missing scores.
    No graph or expression values are written back to the AnnData object.
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, Integral) or n_neighbors < 2:
        raise ValueError("UMAP smoothing requires an integer neighbour count of at least 2.")
    if "X_umap" not in table.obsm:
        raise ValueError("UMAP smoothing requires `X_umap` in the selected table's `obsm`.")
    coordinates = np.asarray(table.obsm["X_umap"], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != table.n_obs or coordinates.shape[1] < 2:
        raise ValueError("`X_umap` must contain at least two coordinates per cell for UMAP smoothing.")
    valid_rows = np.isfinite(coordinates).all(axis=1)
    valid_positions = np.flatnonzero(valid_rows)
    if not len(valid_positions):
        raise ValueError("`X_umap` contains no cells with finite coordinates for UMAP smoothing.")
    if not valid_rows.all():
        logger.warning(
            "{} cells have no finite UMAP coordinates; their module scores will be missing.",
            int((~valid_rows).sum()),
        )
    coordinates = coordinates[valid_rows]
    k = min(int(n_neighbors), len(coordinates))
    entry_count = len(coordinates) * k
    if entry_count > MAX_UMAP_NEIGHBOR_ENTRIES:
        max_k = MAX_UMAP_NEIGHBOR_ENTRIES // len(coordinates)
        raise ValueError(
            f"UMAP smoothing with k={k} over {len(coordinates):,} cells exceeds the viewer's neighbour-memory limit. "
            f"Reduce k to at most {max_k}, or choose an already smoothed layer and disable additional smoothing."
        )
    # Query in bounded batches and build CSR directly. Avoid full distance and
    # repeated-row arrays, which otherwise dominate memory for large k.
    index_dtype = np.int32 if max(table.n_obs, entry_count) <= np.iinfo(np.int32).max else np.int64
    indices = np.empty(entry_count, dtype=index_dtype)
    tree = cKDTree(coordinates)
    batch_size = max(1, min(2048, 1_000_000 // k))
    for start in range(0, len(coordinates), batch_size):
        stop = min(start + batch_size, len(coordinates))
        distances, neighbors = tree.query(coordinates[start:stop], k=k)
        del distances
        neighbors = np.asarray(neighbors).reshape(stop - start, k)
        # Tied coordinates can displace self. Keep exactly k neighbours, including self.
        rows = np.arange(start, stop)
        missing_self = ~(neighbors == rows[:, None]).any(axis=1)
        neighbors[missing_self, -1] = rows[missing_self]
        indices[start * k : stop * k] = valid_positions[neighbors.ravel()]
    indptr = np.zeros(table.n_obs + 1, dtype=index_dtype)
    np.cumsum(valid_rows, dtype=index_dtype, out=indptr[1:])
    indptr *= k
    operator = csr_matrix(
        (
            np.full(entry_count, 1.0 / k, dtype=np.float32),
            indices,
            indptr,
        ),
        shape=(table.n_obs, table.n_obs),
        copy=False,
    )
    return operator, valid_rows


def module_score_values(
    table: AnnData,
    *,
    var_names: Sequence[str],
    obs_positions: np.ndarray,
    layer: str | None = None,
    umap_neighbors: int | None = None,
) -> np.ndarray:
    """Return the mean per-gene z-score for a gene module at ``obs_positions``.

    Genes are read from ``table.X`` or the selected expression layer. Each full
    gene column is optionally smoothed first, standardized across cells with
    population standard deviation (``ddof=0``), and then added to the module
    mean. ``umap_neighbors`` builds neighbours from ``obsm['X_umap']`` only;
    cells with missing UMAP coordinates are excluded and receive NaN scores.
    Constant genes contribute zeros. Columns are processed one at a time,
    keeping peak temporary memory proportional to the number of cells rather
    than cells times genes.
    """
    from napari_harpy.core._color_source import validate_gene_module_genes

    genes = validate_gene_module_genes(var_names, available_var_names=tuple(map(str, table.var_names)))
    obs_positions = np.asarray(obs_positions)
    if obs_positions.ndim != 1 or (
        obs_positions.size
        and (
            not np.issubdtype(obs_positions.dtype, np.integer)
            or np.any(obs_positions < 0)
            or np.any(obs_positions >= table.n_obs)
        )
    ):
        raise ValueError("Could not align linked table rows to the gene-module score for viewer coloring.")
    obs_positions = obs_positions.astype(np.intp, copy=False)

    matrix = resolve_expression_matrix(table, layer)
    if matrix is None:
        raise ValueError("The selected expression matrix is empty; choose a populated table layer.")
    smoother = None
    valid_rows = np.ones(table.n_obs, dtype=bool)
    if umap_neighbors is not None:
        smoother, valid_rows = _umap_neighbor_operator(table, umap_neighbors)

    score = np.zeros(table.n_obs, dtype=np.float64)
    for gene in genes:
        var_index = table.var_names.get_loc(gene)
        if not isinstance(var_index, (int, np.integer)):
            raise ValueError(f"Var `{gene}` does not resolve to one unique column in `table.var_names`.")
        column = matrix[:, int(var_index)]
        column = column.toarray().reshape(-1) if hasattr(column, "toarray") else np.asarray(column).reshape(-1)
        column = np.asarray(column, dtype=np.float64)
        if not np.all(np.isfinite(column[valid_rows])):
            raise ValueError(f"Var `{gene}` contains non-finite values and cannot be used in a gene-module score.")
        if smoother is not None:
            column = np.asarray(smoother @ column).reshape(-1)
        valid_column = column[valid_rows]
        if not valid_column.size:
            continue
        if valid_column.max() == valid_column.min():
            continue  # Constant genes contribute zero, even when mean summation rounds slightly.
        mean = valid_column.mean(dtype=np.float64)
        standard_deviation = valid_column.std(dtype=np.float64)
        if not np.isfinite(mean) or not np.isfinite(standard_deviation):
            raise ValueError(f"Var `{gene}` produced non-finite values while computing its gene-module score.")
        if standard_deviation != 0:
            score[valid_rows] += (valid_column - mean) / standard_deviation

    score /= len(genes)
    score[~valid_rows] = np.nan
    return score[obs_positions]


def categorical_colors_for_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[str],
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
) -> pd.Series:
    lookup = {normalize_category_value(category): color for category, color in zip(categories, palette, strict=False)}
    colors = {}
    for index, value in values.items():
        if pd.isna(value):
            colors[index] = missing_color
            continue
        colors[index] = lookup.get(normalize_category_value(value), missing_color)
    return pd.Series(colors, index=values.index, dtype="object")


def categorical_rgba_for_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[str],
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
) -> np.ndarray:
    """Return one RGBA color row per categorical value."""
    rgba = np.empty((len(values), 4), dtype="float64")
    rgba[:] = to_rgba(missing_color)
    if len(values) == 0:
        return rgba

    normalized_categories = [normalize_category_value(category) for category in categories]
    palette_rgba = np.asarray(
        [to_rgba(color) for _, color in zip(normalized_categories, palette, strict=False)],
        dtype="float64",
    )
    if len(palette_rgba) == 0:
        return rgba

    category_code_by_value = {
        category: code for code, category in enumerate(normalized_categories[: len(palette_rgba)])
    }
    if isinstance(values.dtype, pd.CategoricalDtype):
        value_categories = [normalize_category_value(category) for category in values.cat.categories]
        palette_code_by_value_code = np.asarray(
            [category_code_by_value.get(category, -1) for category in value_categories],
            dtype=np.int64,
        )
        value_codes = values.cat.codes.to_numpy(copy=False)
        present_values = value_codes >= 0
        if np.any(present_values):
            palette_codes = palette_code_by_value_code[value_codes[present_values]]
            known_values = palette_codes >= 0
            present_indices = np.flatnonzero(present_values)
            rgba[present_indices[known_values]] = palette_rgba[palette_codes[known_values]]
        return rgba

    normalized_values = pd.Series(
        [pd.NA if pd.isna(value) else normalize_category_value(value) for value in values],
        index=values.index,
        dtype="object",
    )
    palette_codes = normalized_values.map(category_code_by_value)
    known_values = palette_codes.notna().to_numpy(dtype=bool, copy=False)
    if np.any(known_values):
        rgba[known_values] = palette_rgba[palette_codes.loc[known_values].to_numpy(dtype=np.int64, copy=False)]
    return rgba


def continuous_colors_for_values(
    values: pd.Series,
    *,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    colormap_name: str | None = None,
) -> pd.Series:
    colors = {}
    non_missing = values.dropna()
    if non_missing.empty:
        for index in values.index:
            colors[index] = missing_color
        return pd.Series(colors, index=values.index, dtype="object")

    cmap = colormaps[resolve_continuous_colormap(colormap_name)]
    min_value = float(non_missing.min())
    max_value = float(non_missing.max())
    if max_value == min_value:
        normalized_values = dict.fromkeys(non_missing.index, 0.5)
    else:
        normalized_values = {
            index: float((value - min_value) / (max_value - min_value)) for index, value in non_missing.items()
        }

    for index in values.index:
        value = values.at[index]
        if pd.isna(value):
            colors[index] = missing_color
        else:
            colors[index] = cmap(float(np.clip(normalized_values[index], 0.0, 1.0)))
    return pd.Series(colors, index=values.index, dtype="object")


def continuous_rgba_for_values(
    values: pd.Series,
    *,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    colormap_name: str | None = None,
) -> np.ndarray:
    """Return one RGBA color row per continuous value."""
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", copy=False)
    rgba = np.empty((len(value_array), 4), dtype="float64")
    rgba[:] = to_rgba(missing_color)
    present_values = ~np.isnan(value_array)
    if not np.any(present_values):
        return rgba

    non_missing = value_array[present_values]
    cmap = colormaps[resolve_continuous_colormap(colormap_name)]
    min_value = float(non_missing.min())
    max_value = float(non_missing.max())
    if max_value == min_value:
        normalized_values = np.full(len(non_missing), 0.5, dtype="float64")
    else:
        normalized_values = np.clip((non_missing - min_value) / (max_value - min_value), 0.0, 1.0)
    rgba[present_values] = cmap(normalized_values)
    return rgba


def is_valid_color(value: str) -> bool:
    try:
        to_rgba(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_string_scalar(value: object) -> bool:
    return isinstance(value, (str, bytes, np.str_, np.bytes_))
