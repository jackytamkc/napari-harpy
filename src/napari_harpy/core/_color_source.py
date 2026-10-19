from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, cast

TableColorSourceKind = Literal["obs_column", "x_var", "x_module"]
TableColorValueKind = Literal["categorical", "continuous", "instance"]
ShapeColorSourceKind = Literal["shape_column"]
ShapeColorValueKind = Literal["categorical", "continuous"]
TABLE_COLOR_SOURCE_KINDS: tuple[TableColorSourceKind, ...] = ("obs_column", "x_var", "x_module")
TABLE_COLOR_VALUE_KINDS: tuple[TableColorValueKind, ...] = ("categorical", "continuous", "instance")
SHAPE_COLOR_SOURCE_KINDS: tuple[ShapeColorSourceKind, ...] = ("shape_column",)
SHAPE_COLOR_VALUE_KINDS: tuple[ShapeColorValueKind, ...] = ("categorical", "continuous")


def validate_table_color_source_kind(kind: str) -> TableColorSourceKind:
    """Return a validated table color source kind."""
    if kind not in TABLE_COLOR_SOURCE_KINDS:
        raise ValueError(_format_invalid_kind_error("table color source kind", kind, TABLE_COLOR_SOURCE_KINDS))

    return cast(TableColorSourceKind, kind)


def validate_table_color_value_kind(kind: str) -> TableColorValueKind:
    """Return a validated table color value kind."""
    if kind not in TABLE_COLOR_VALUE_KINDS:
        raise ValueError(_format_invalid_kind_error("table color value kind", kind, TABLE_COLOR_VALUE_KINDS))

    return cast(TableColorValueKind, kind)


def validate_shape_color_source_kind(kind: str) -> ShapeColorSourceKind:
    """Return a validated shape color source kind."""
    if kind not in SHAPE_COLOR_SOURCE_KINDS:
        raise ValueError(_format_invalid_kind_error("shape color source kind", kind, SHAPE_COLOR_SOURCE_KINDS))

    return cast(ShapeColorSourceKind, kind)


def validate_shape_color_value_kind(kind: str) -> ShapeColorValueKind:
    """Return a validated shape color value kind."""
    if kind not in SHAPE_COLOR_VALUE_KINDS:
        raise ValueError(_format_invalid_kind_error("shape color value kind", kind, SHAPE_COLOR_VALUE_KINDS))

    return cast(ShapeColorValueKind, kind)


@dataclass(frozen=True)
class TableColorSourceSpec:
    """Semantic description of one table-backed source used for viewer coloring."""

    table_name: str
    source_kind: TableColorSourceKind
    value_key: str
    value_kind: TableColorValueKind
    # Expression layer for ``x_var`` and ``x_module`` sources: ``None`` (or "X") reads
    # ``table.X``; any other value reads ``table.layers[layer]`` (e.g. "log1p", "raw",
    # "smoothed"). Ignored for obs sources.
    layer: str | None = None
    # Optional cell-graph smoother for single-var sources: when set, the gene column is averaged
    # over each cell's neighbourhood live via ``table.obsp[smoother]`` (e.g. "smoother_k100").
    # ``None`` = unsmoothed.
    smoother: str | None = None
    # Matplotlib colormap name for *continuous* coloring (e.g. "viridis", "inferno", "magma"). ``None``
    # falls back to the viridis default. Ignored for categorical/instance sources (they use palettes).
    colormap: str | None = None
    # Ordered genes used by an ``x_module`` source. The displayed module value is the mean of
    # the per-gene z-scores from the selected expression matrix. Empty for every other source.
    module_genes: tuple[str, ...] = ()
    # Build a uniform k-neighbour smoother from table.obsm['X_umap'] for a module.
    # Includes the cell itself; None reads the selected layer without additional UMAP smoothing.
    umap_neighbors: int | None = None

    def __post_init__(self) -> None:
        validate_table_color_source_kind(self.source_kind)
        validate_table_color_value_kind(self.value_kind)
        if self.source_kind == "x_module":
            module_genes = validate_gene_module_genes(self.module_genes)
            object.__setattr__(self, "module_genes", module_genes)
            object.__setattr__(self, "value_key", gene_module_value_key(module_genes))
            if self.value_kind != "continuous":
                raise ValueError("Gene-module color sources must use `value_kind='continuous'`.")
            if self.smoother is not None:
                raise ValueError(
                    "For gene modules, use `umap_neighbors` for UMAP smoothing or None for the selected layer only; "
                    "`smoother` is unsupported."
                )
            if self.umap_neighbors is not None:
                if (
                    isinstance(self.umap_neighbors, bool)
                    or not isinstance(self.umap_neighbors, Integral)
                    or self.umap_neighbors < 2
                ):
                    raise ValueError("UMAP smoothing requires an integer neighbour count of at least 2.")
                object.__setattr__(self, "umap_neighbors", int(self.umap_neighbors))
        elif self.module_genes:
            raise ValueError("`module_genes` can only be set for an `x_module` color source.")
        elif self.umap_neighbors is not None:
            raise ValueError("`umap_neighbors` can only be set for an `x_module` color source.")

    @property
    def identity(self) -> tuple[str, TableColorSourceKind, str]:
        """Return a stable identity for preserving selection across refreshes."""
        return (self.table_name, self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        """Return the default user-facing name for this source."""
        return self.value_key


@dataclass(frozen=True)
class ShapeColumnColorSourceSpec:
    """Semantic description of one direct shapes column used for coloring."""

    source_kind: ShapeColorSourceKind
    value_key: str
    value_kind: ShapeColorValueKind
    # Matplotlib colormap name for *continuous* coloring (e.g. "viridis", "inferno", "magma"). ``None``
    # falls back to the viridis default. Ignored for categorical sources (they use palettes).
    colormap: str | None = None

    def __post_init__(self) -> None:
        validate_shape_color_source_kind(self.source_kind)
        validate_shape_color_value_kind(self.value_kind)

    @property
    def identity(self) -> tuple[ShapeColorSourceKind, str]:
        """Return a stable identity for preserving selection across refreshes."""
        return (self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        """Return the default user-facing name for this source."""
        return self.value_key


def _format_invalid_kind_error(kind_label: str, kind: str, allowed_kinds: tuple[str, ...]) -> str:
    allowed = ", ".join(repr(allowed_kind) for allowed_kind in allowed_kinds)
    return f"Invalid {kind_label} {kind!r}. Expected one of: {allowed}."


def parse_gene_module_text(text: str) -> tuple[str, ...]:
    """Parse a comma-, semicolon-, or whitespace-separated gene list."""
    return tuple(gene for gene in re.split(r"[,;\s]+", text.strip()) if gene)


def validate_gene_module_genes(
    genes: Sequence[str],
    *,
    available_var_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return a validated ordered gene module.

    A module needs at least two distinct, non-empty gene names. When
    ``available_var_names`` is supplied, every module gene must also exist in
    the selected table.
    """
    if isinstance(genes, str):
        raise ValueError("A gene module requires a sequence of gene names, not one string.")

    normalized = tuple(str(gene).strip() for gene in genes)
    if any(not gene for gene in normalized):
        raise ValueError("Gene-module names cannot be empty.")
    if len(normalized) < 2:
        raise ValueError("A gene module requires at least two genes.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for gene in normalized:
        if gene in seen and gene not in duplicates:
            duplicates.append(gene)
        seen.add(gene)
    if duplicates:
        raise ValueError(f"Remove duplicate module genes: {_format_gene_preview(duplicates)}.")

    if available_var_names is not None:
        available = set(available_var_names)
        missing = [gene for gene in normalized if gene not in available]
        if missing:
            raise ValueError(f"Module genes not found in the selected table: {_format_gene_preview(missing)}.")
    return normalized


def gene_module_value_key(genes: Sequence[str]) -> str:
    """Return the stable feature/layer label for an ordered module gene list."""
    return " + ".join(validate_gene_module_genes(genes))


def _format_gene_preview(genes: Sequence[str]) -> str:
    preview = ", ".join(genes[:5])
    return f"{preview}, ..." if len(genes) > 5 else preview
