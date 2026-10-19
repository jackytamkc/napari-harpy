# napari-harpy fork — changes vs upstream

Every change in this fork relative to upstream, grouped so you can decide **what to PR and what to
keep local**. Maintained as work continues.

| | |
|---|---|
| Fork | `jackytamkc/napari-harpy` · clone `/home/beast/napari-harpy-fork` |
| Branch | `feat/category-colors-and-sopa-fixes` |
| Base | `c832868` — identical to `upstream/vibspatial/napari-harpy@main` (no commits on the branch) |
| State | **everything below is uncommitted in the working tree** (no commits, no PR) |
| Env | editable-installed into conda env `harpy` |

**Regenerate the raw diff at any time:**
```bash
cd /home/beast/napari-harpy-fork
git diff                              # all modified tracked files vs upstream
git status --short                    # modified (M) + new (??) files
git ls-files --others --exclude-standard | grep -v __pycache__   # just the new files
```

**Related narrative log:** `/home/beast/sopa/Liver_spatial_Ramachandran/harpy/HARPY_FINDINGS.md`
(the story + root causes + code-review history). This file is the terse, PR-oriented index.

---

## Summary

| # | Change | Type | New? | Reviewed | Files (primary) |
|---|---|---|---|---|---|
| 1 | Table-backed shapes styling accepts an **unnamed** GeoDataFrame index | Bug (interop) | — | ✔ | `viewer/shapes_styling.py` |
| 2 | Faster spawn — defer the transcript scan until **Points** is expanded | Perf | — | ✔ | `widgets/viewer/widget.py` |
| 3 | **Per-category colour editor** (swatch + visibility toggle) | Feature | ✚ | ✔ | `viewer/category_colors.py`, `widgets/viewer/category_color_editor.py`, widgets, styling |
| 4 | Colour by a **gene from any expression layer** (+ optional UMAP smoother, + **continuous colormap choice**) | Feature | — | ✔ (+ colormap round) | `core/_color_source.py`, `viewer/_styling.py`, styling, widgets |
| 5 | **obs selector → scrollable dropdown** (not free-text) | UX | — | ✔ | `widgets/viewer/{shapes,labels}_widget.py` |
| 6 | **Viewer presets** — save/restore display (incl. **layer stacking order**), autoload by channel panel | Feature | ✚ | ✔ (3 rounds + order fix) | `viewer/viewer_config.py`, `_interactive.py` |
| 7 | loguru format-string fix (`%s` → `{}`) | Fix (1 line) | — | ✔ | `viewer/adapter.py` |
| 8 | **Hiding a category survives *opaque* blending** — shapes (exclude geometry / `shown`) **and labels (mask hidden IDs to background, display-only)** | Bug (rendering) | — | ✔ (2 rounds + labels) | `viewer/adapter.py`, `viewer/shapes_styling.py`, `viewer/labels_styling.py`, `viewer/category_colors.py` |
| 9 | **Gene modules with UMAP-neighbour smoothing** for labels and shapes, saved in presets | Feature | ✚ | ✔ (2 rounds) | `core/_color_source.py`, `viewer/_styling.py`, styling, widgets, presets |

"New? ✚" = self-contained new module (cleanest to PR alone). The rest are edits interleaved into shared
files — see **Entanglement** at the bottom before splitting a PR.

---

## 1. Unnamed-index shapes styling (interop bug)

**What:** upstream refused to style shapes from a linked table unless `shapes.index.name == instance_key`.
SpatialData readers (sopa, spatialdata-io) leave shape indices unnamed, so standard objects tripped it.
Now an unnamed index is accepted (its *values* are used as the instance identity, matching SpatialData's
own value-based join); only an index explicitly named *something else* is rejected.

**Files:** `viewer/shapes_styling.py` — `_get_shape_instance_values` (hunk `@@ -491`).

**PR-ready:** yes, and the most isolated of the in-file edits. Self-contained to one function.

---

## 2. Deferred transcript scan on spawn (performance)

**What:** upstream eagerly ran a full Dask scan of the transcripts on viewer spawn (~1 s on a CosMX
sample). Now the scan is deferred until the user actually expands the **Points** section. Spawn on
P3_MASLD2 dropped ~1317 ms → ~927 ms.

**Files:** `widgets/viewer/widget.py` — `_on_points_section_toggled`, `_load_points_value_source_if_pending`,
and the removal of the eager `load_value_source()` from the points bind.

**PR-ready:** yes in principle, but lives in `widget.py` alongside features 3–5 (see Entanglement).

---

## 3. Per-category colour editor (new feature)

**What:** a non-modal dialog to set an explicit colour per category and **toggle a category's
visibility**, persisted to the scanpy-standard `table.uns[f"{col}_colors"]`. Hidden = the palette entry
is written with **alpha 0**.

**New files:**
- `viewer/category_colors.py` (120 ln) — framework-free engine: `uns_colors_key`, `category_values`,
  `get_category_palette`, `store_category_colors`, `reset_category_colors`, `storage_hex`.
- `widgets/viewer/category_color_editor.py` (271 ln) — the Qt `CategoryColorEditor(QDialog)`
  (per-category swatch + visibility checkbox, Show all / Hide all, `colors_changed` signal,
  `WA_DeleteOnClose`).

**Edits in shared files:**
- `widgets/viewer/widget.py` — editor wiring: `_edit_shapes_category_colors`,
  `_edit_labels_category_colors`, `_open_category_color_editor`, `_find_open_category_color_editor`,
  `_forget_category_color_editor`, `_close_open_category_color_editors` (dedup + close-on-sdata-change).
- `viewer/shapes_styling.py` — `_copy_rgba_with_alpha` (hunk `@@ -1087`) preserves alpha-0 rows so a
  hidden category stays hidden through re-styling.
- `viewer/labels_styling.py` — the labels colormap renders an alpha-0 category transparent.
- `widgets/viewer/{shapes,labels}_widget.py` — the "Edit colours…" button + `_emit_edit_colors_request`.

**Tests:** `tests/test_category_colors.py` (90 ln), `tests/test_category_color_editor.py` (115 ln).

**Known limitation (being fixed by #8):** hiding via alpha 0 works in translucent blending but **not
opaque** (napari's opaque blending sets `blend: False`, ignoring alpha).

---

## 4. Colour by a gene from any expression layer + UMAP smoother (feature)

**What:** upstream could colour shapes/labels by a gene only from `table.X`. Now any expression layer
(`log1p`, `raw`, `smoothed`, …) is selectable, plus an optional on-the-fly cell-neighbourhood smoother
read live from `table.obsp[...]`.

**Files:**
- `core/_color_source.py` — `TableColorSourceSpec` gains `layer: str | None` and `smoother: str | None`
  (hunk `@@ -53`; in `__eq__`, **not** in `identity`).
- `viewer/_styling.py` — new `resolve_expression_matrix(table, layer)` and
  `var_expression_values(table, *, var_name, obs_positions, layer, smoother_key)` (reads the full gene
  column, then `(A @ column)[obs_positions]` when a smoother is set).
- `viewer/labels_styling.py` — `_build_x_var_colormap(..., layer, smoother)`; `build_styled_labels_layer_name`
  encodes `layer~smoother`.
- `viewer/shapes_styling.py` — `_get_table_color_values_by_instance` passes `layer`/`smoother`;
  `build_styled_shapes_layer_name` encodes `layer~smoother`.
- `widgets/viewer/widget.py` — `_table_expression_layers`, `_table_smoothers` (enumerate layers/obsp).
- `widgets/viewer/{shapes,labels}_widget.py` — a **Layer** combo + **Smooth over UMAP neighbours**
  checkbox, and `_selected_layer`, `_selected_smoother`, `_set_layer_row_visible`, `_set_smooth_row_visible`.

**Tests:** additions in `tests/test_styling.py` (+115), `tests/test_shapes_styling.py` (+102),
`tests/test_viewer_widget.py` (+402, shared with 3 & 5).

**Continuous colormap choice (2026-07-17):** continuous coloring (gene / numeric obs / numeric shape
column) was hardcoded to viridis; the user can now pick any matplotlib colormap (inferno, magma, …).
A `colormap: str | None` field on both specs (`None` → viridis; in `__eq__` so a change re-styles, out
of `identity` so it doesn't fork the layer — like `layer`/`smoother`) threads from a new "Colourmap"
dropdown to the two leaf color builders (`continuous_rgba_for_values` / `_build_continuous_shape_style`
/ `_build_continuous_color_dict`). New `CONTINUOUS_COLORMAP_CHOICES` + `resolve_continuous_colormap`
(`viewer/_styling.py`) fall back to viridis for `None`/stale/invalid names so a bad preset never raises.
The dropdown shows only for continuous sources (hidden for categorical/instance) and returns `None` for
the default viridis to keep specs/presets minimal. Persisted via `_serialize/_deserialize_style_spec`
(`viewer_config.py`, backward-compatible: old presets → `None`). **Files:** `core/_color_source.py`
(both spec dataclasses), `viewer/_styling.py`, `viewer/shapes_styling.py` (`_build_continuous_shape_style`
+ `_build_table_obs_shape_style` + 3 call sites), `viewer/labels_styling.py` (`_build_continuous_color_dict`
+ `_build_x_var_colormap` + `_build_obs_column_colormap` + dispatch), `viewer/viewer_config.py`,
`widgets/viewer/{shapes,labels}_widget.py` (Colourmap combo, `_selected_colormap`, `_set_colormap_row_visible`).
**Tests:** `test_styling.py` (resolver, leaf + color_dict honor the colormap), `test_viewer_config.py`
(spec round-trips colormap; legacy preset → None), `test_viewer_widget.py` (injected for continuous only,
row hidden for categorical). End-to-end verified through the adapter: viridis vs inferno produce different
`DirectLabelColormap` colors on the same instances.

**Reuse-in-place fix (code review, critical).** First review round caught that a colormap change *forked
a duplicate styled layer* instead of re-styling: `colormap` is in the spec `__eq__` (the adapter's
reuse-match key via `_matches_labels_binding`/`_matches_shapes_binding`) but NOT in the layer name, so a
colormap-only change failed the match → built a second layer with the *same* name, leaving the old
wrong-colored one behind (and both got captured into presets). Unlike `layer`/`smoother`, which *are* in
the name and legitimately fork. Fixed in `viewer/adapter.py`: the reuse match now compares specs via
`_style_specs_match_ignoring_display` (equal apart from `colormap`), and the reuse path calls the new
`LayerBindingRegistry.update_binding_style_spec` to refresh the frozen binding's spec so preset capture /
later matches see the current colormap. Verified: viridis→inferno→magma→inferno stays at **1 layer** for
both labels and shapes, binding tracks the current colormap, layer is genuinely re-colored. +2 adapter
regression tests (`test_..._restyles_in_place_on_colormap_change` for labels and shapes). The
`test_styled_shapes_hiding.py` #8 reuse-vs-rebuild suite still passes (the hidden-signature rebuild is
colormap-independent). For consistency, `LayerBindingRegistry.find_bindings(style_spec=...)` was aligned
to the same colormap-ignoring match (it had no production callers but was a latent trap). **Second review:
APPROVED** — the reviewer independently reproduced the 3-layers→1-layer fix for labels and shapes,
confirmed the shared-matcher change is safe for every caller, that frozen-`replace` preserves all binding
fields, that a simultaneous hidden-set + colormap change still rebuilds, and that the 2 regression tests
fail on pre-fix code.

---

## 5. obs selector → scrollable dropdown (UX)

**What:** the "colour by observation" selector used to require typing the column name. obs columns are
few and researcher-named, so it is now a non-editable dropdown (genes stay typed with a completer). The
value selector is a `QStackedWidget` switching between a combo (obs) and a line-edit+completer (genes /
shape columns).

**Files:** `widgets/viewer/{shapes,labels}_widget.py` — `_current_value_text`, the `QStackedWidget`
value selector; `widgets/viewer/widget.py` — request plumbing.

**PR-ready:** conceptually simple, but the same widget files/methods carry features 3 & 4.

---

## 6. Viewer presets (new feature)

**What:** capture how a sample is displayed (image channel colormap/contrast/gamma/blending/opacity/
visibility, styled labels/shapes colour source, per-category palettes, coordinate system, scale bar,
camera, **layer stacking order**, best-effort points selection) into JSON and re-apply it —
automatically on open for any sample sharing the same **image channel panel**. Shift-S save, Shift-D apply.

**New file:** `viewer/viewer_config.py` (865 ln) — `ViewerConfigManager` + `get_preset_dir`
(`~/.napari-harpy/viewer_presets/`, env `NAPARI_HARPY_PRESET_DIR`).

**Edits:** `_interactive.py` — `Interactive(..., autoload=True)` + `save_config` / `load_config` /
`list_configs` + the Shift-S/Shift-D keybinds (hunk `@@` in `__init__`).

**Tests:** `tests/test_viewer_config.py` (39 tests).

**PR-ready:** the most self-contained feature — `viewer_config.py` is new and `_interactive.py`'s edits
are additive and touch nothing else. Cleanest single-feature PR. Went through **3 code-review rounds**
(see HARPY_FINDINGS.md Feature 5); all findings fixed and mutation-verified.

**Layer-order fix (2026-07-16):** on reload, layers came back in the wrong z-order — `_plan_layers`
rebuilds overlay images *last*, so images landed on top and buried the styled shapes/labels
(`final_boundaries[obs:scNiche]`) the user had dragged on top. Now `capture()` records the live
`viewer.layers` order (bottom→top) under `"layer_order"` via `_capture_layer_order()`, and `apply()`
replays it with `_restore_layer_order()` (`viewer.layers.move`) right after building the layers. Matches
layers by name; missing names (a layer that failed to load) are skipped and un-named live layers float
above, stably. Old presets lack the field, and a malformed one is ignored (`None`/non-list/non-str
entries → no-op, matching the sibling `_restore_palettes` corrupt-file resilience). +6 tests (4
reorder-mechanics unit tests over a **napari-faithful** fake LayerList, 1 malformed-input guard, 1
full-stack save/reload round-trip); 40 total in `test_viewer_config.py`. This is a napari-side ordering
fix, independent of the #8 single-layer z-order-across-rebuild preservation. **Code-reviewed** (1 round):
the reorder loop was brute-forced correct against the real `LayerList` over all n≤4 permutations; the one
finding (missing malformed-input guard → `TypeError` could abort `apply()`) is fixed above.

---

## 7. loguru format-string fix

**What:** `viewer/adapter.py` used a printf `%s` placeholder in a `logger.warning` — loguru uses
`str.format`, so the layer name never interpolated. Changed `%s` → `{}`. **One line.** Trivially PR-able.

---

## 8. Hiding a category survives opaque blending (styled shapes)

**What:** a category hidden in the colour editor (palette alpha 0) reappeared under napari's **opaque**
blending, which disables GL blending (`blend: False` in `napari/_vispy/utils/gl.py`) and ignores alpha.
Now hiding is blending-independent for styled shapes:
- **Polygon shapes** (`_HarpyShapes`): the hidden categories' **geometry is excluded** from the layer
  (napari Shapes has no per-shape visibility). The styled layer is rebuilt with only visible source
  rows when the hidden set changes; unchanged sets reuse the layer.
- **Point-radius shapes** (circles → `_HarpyPointRadiusShapes(Points)`): geometry stays full, hidden
  via napari's per-point **`layer.shown`** mask (honoured regardless of blending).

**Files:**
- `viewer/category_colors.py` — `hidden_categories(table, column)` → the categories stored with alpha 0.
- `viewer/shapes_styling.py` — `source_row_hidden_mask(sdata, shapes_name, style_spec)` → per-source-row
  hidden/unassigned mask (categorically rendered obs table sources only; `None` when nothing is masked).
- `viewer/adapter.py` — `_prepare_napari_shapes_layer_inputs(..., included_source_rows=)` skips hidden
  source rows (kept out of `data`/`features`/mapping together; **not** counted as
  `skipped_geometry_count`, which gates annotation editing); `_build_shapes_layer(..., included_source_rows=)`;
  the `ensure_styled_shapes_loaded` reuse-vs-rebuild logic + `_STYLED_SHAPES_HIDDEN_SIGNATURE_KEY` +
  `_apply_point_radius_hidden_visibility`.

**Tests:** `tests/test_styled_shapes_hiding.py` (22) — exclusion, opaque-still-hidden, unhide restores,
hide-all → empty, MultiPolygon parts all removed, point-radius `shown`, reuse-vs-rebuild, unassigned
categorical values (including absent table rows), assignment-change rebuilds, continuous exclusion,
live-render-dtype gating, and the view-state-preservation-across-rebuild regression. Each core mechanism
is mutation-verified (including the z-order restore). Two `code-reviewer` rounds on the original hiding
fix: round 1 caught a real regression (a rebuild discarded the user's visible/opacity/blending/z-order
— now captured and restored); round 2 caught that the z-order line wasn't actually exercised by its
test (fixed: the test now places the styled layer below another so the reorder genuinely runs).

A geometry rebuild (on a hidden-set change) carries the user's `visible`/`opacity`/`blending`/z-order
across the new layer, so hiding a second category can't reset a deliberately-opaque styled layer
(a regression caught in review, now fixed + tested).

**Categorically unassigned Shapes now covered too (2026-07-23).** `source_row_hidden_mask` also masks
rows whose aligned obs value is missing when that live obs dtype takes
`_build_table_obs_shape_style`'s categorical branch. This covers both an explicit `NaN` cell type and
a shape with no linked table row, so the flat `#808080` missing colour cannot remain after hiding every
real type. Polygon geometry is omitted; point-radius rows use `shown=False`. As with Labels, the gate
mirrors the live renderer rather than trusting a potentially stale `value_kind`; continuous missing
values retain their previous behavior. The mask's actual source-row positions remain the rebuild
signature, so assigning or unassigning a cell is reflected on the next restyle.

**Styled LABELS now covered too (2026-07-22).** The originally-deferred labels case is fixed: for a
styled labels layer coloured by a categorical obs column, hidden categories' **label IDs are remapped to
background (0) in the DISPLAYED raster** (napari's opaque blending ignores the `DirectLabelColormap`
alpha, so masking the data is the only blending-independent hide). **Unassigned (NaN-category) cells are
masked the same way (2026-07-22b):** they otherwise render the flat grey `#808080` missing colour and
persisted after "hide all types" (a user hit exactly this — 7.7% of a MASLD TMA figure was grey untyped
cells). `masked_label_ids` adds `pd.isna` cells for a column that **renders categorically** (gated by
`_obs_column_renders_categorical`, which mirrors `_build_obs_column_colormap`'s branch selection — NOT
`value_kind`, which the classifier sets differently and can be stale in a preset; e.g. an all-0/all-1
nullable-int column renders categorical but is classified continuous). So an untyped cell is treated as
"not a visible type" and hiding every type gives a clean overlay; continuous coloring is unaffected.
Code-reviewed (round found: `value_kind` gate diverged from the render for binary-int columns → fixed
via the render predicate; a dead `hidden_label_ids` twin → removed, `masked_label_ids` is the single
impl; NaN check vectorised; +continuous-exclusion, binary-int, and multiscale-unassigned regression
tests). `test_styled_labels_hiding.py` now 16 tests. It is **display-only — the element /
on-disk zarr is never modified** (the user's explicit constraint): `_remap_labels_to_background`
(`viewer/adapter.py`) unwraps each level's xarray `DataArray` to its underlying array and masks it with
a per-block `np.isin` — a **lazy `da.map_blocks` on the dask array** (the full raster is never
materialised), rewrapped to keep dims/coords. It dispatches on **`layer.multiscale`** (napari's
multiscale `layer.data` is a `MultiScaleData`, *not* a list) and masks every pyramid level.
`_apply_labels_hidden_ids` stashes the original `layer.data` on `layer.metadata` so a hidden-set change
re-masks from source (never remask-on-remask) and unhide restores. The change key is the **actual hidden
label-ID set** (`frozenset(hidden_label_ids(...))`) — faithful to BOTH the hidden categories AND the obs
cell→category assignments, so reassigning a cell into an already-hidden class is caught; `hidden_label_ids`
short-circuits before the obs scan when nothing is hidden, keeping the common case cheap. Tests:
`tests/test_styled_labels_hiding.py` (10) — hide drops IDs from display but not the element, stays gone
under opaque, unhide restores, hide-all → all background, zarr untouched, reassign-into-hidden-class
re-masks, **plus a pyramided-multiscale case and a dask-stays-lazy case**. Runs on both the created and
reused styled-labels paths. **Code-reviewed (3 rounds):** round 1 REJECTED — `isinstance(list/tuple)`
missed `MultiScaleData` (pyramided labels crashed) and `np.asarray` on the `DataArray` eagerly
materialised the full raster; round 2 caught that a perf-optimised uns-only change key silently skipped
the re-mask on an obs reassignment (stale display) — reverted to the faithful ID-based key. All fixed
and pinned by the new tests; multiscale + laziness + zarr-safety independently re-verified.

**Known limitations / deferred (from code review):**
- Unannotated shapes are excluded only for categorically rendered obs overlays. Continuous obs/gene
  overlays and shape-column sources retain their existing missing-value semantics.
- **Shape-column** and continuous colour sources have no hiding UI, so they are unaffected. The
  pre-existing `_copy_rgba_with_alpha` alpha-0 preservation (from change #3) is now vestigial for the
  table path (#8 supersedes it via exclusion/`shown`); it is kept because a dedicated #3 test pins it.
- **Perf:** when a category is hidden, the table alignment runs twice per re-style (once to compute the
  hidden mask before build, once to compute colours after). The no-hidden common case is unaffected
  (an early return). Deferred; could be de-duplicated by threading one alignment through.

---

## 9. Gene modules with UMAP smoothing (2026-09-24)

Choose **Color source → Gene module**, enter at least two comma-separated gene names, and select
an expression **Layer** (normally `X` or `log1p`). **UMAP neighbours** is enabled by default with
`k=100`; the neighbour count is adjustable. Click **Add / Update in viewer**. Available for both
Labels and Shapes, including point-radius shapes; the gene list, layer, UMAP setting and colormap
are included in viewer presets.

The score is the mean of per-gene population z-scores **after** smoothing each selected gene over
its k nearest cells in `table.obsm['X_umap']`, including self. Neighbours and z-score normalization
use the selected linked table. No tissue/spatial-coordinate graph or unnamed `obsp` graph is used
for modules. Missing/duplicate genes fail explicitly. Constant genes contribute zero.

For an already smoothed matrix, select that layer and **uncheck UMAP neighbours** to avoid applying
smoothing twice. The user's `smoothed_k100` layer was generated by
`/home/beast/cosmx/umap_smooth_k100.py` using UMAP k=100 over the combined dataset. Live smoothing
on a per-sample zarr uses that sample's available cells, so the neighbourhoods can differ from the
precomputed combined-dataset layer. Scores and colormap ranges are relative to the selected table;
this does not establish a shared cross-sample scale.

Cells without finite UMAP coordinates are excluded from the live neighbour graph and z-score
normalization, and receive NaN module scores (the standard **grey missing-value color** for continuous
overlays). A warning reports the count. There is no spatial-coordinate fallback or coordinate
imputation. At least one finite UMAP row is required; k is capped to the number of valid cells.

Smoothing reads one gene column at a time. kNN queries are batched and the CSR graph uses compact
indices/weights. An explicit 20-million-neighbour-entry limit rejects excessive allocation with an
actionable error. The graph and scores are display-only; no `obs`, `obsp`, expression layer or zarr
is written. Fresh styled layers are removed on failure, and a failed restyle retains the previously
successful binding, avoiding leftover unstyled layers after a cross-sample preset fails.

**Tests:** numerical UMAP-vs-spatial neighbour selection, finite-row handling, self-inclusion with
tied coordinates, constant genes, memory guard, input validation, UI state/dispatch, preset fields,
and integration for labels, polygons and circles (including failure cleanup). The real
`P3_MASLD2/processed_table` collagen module was checked on all 81,855 rows: 62,753 valid UMAP scores
and 19,102 missing scores. Existing points-panel minimum-width test failure predates this change.
Final regression run: **422 passed**, with only that same pre-existing width assertion failing;
the final compact module-control layout also passed the focused UI checks. Independent code review
cleared the UMAP computation, alignment, memory bound, preset persistence and failure cleanup.

**Environment:** implemented in the editable `/home/beast/napari-harpy-fork` imported by the `harpy`
kernel used in `/home/beast/Thesis/harpy.ipynb`. Restart that kernel and relaunch the viewer to load
the added controls.

---

## Entanglement — read before splitting PRs

The **new-module** changes are cleanly separable and can each be PR'd alone:
`viewer/viewer_config.py` (6), `viewer/category_colors.py` + `widgets/viewer/category_color_editor.py`
(3), and the one-line `adapter.py` fix (7).

The **in-file edits for features 2, 3, 4, 5 are interleaved** in the same handful of files, so they are
not cleanly separable per-file:

| File | Features touching it |
|---|---|
| `widgets/viewer/widget.py` | 2 (deferred points), 3 (editor wiring), 4 (layer/smoother enum), 5 (dropdown plumbing) |
| `widgets/viewer/shapes_widget.py` | 3 (edit-colours button), 4 (layer/smoother combos), 5 (obs dropdown) |
| `widgets/viewer/labels_widget.py` | 3, 4, 5 (same as shapes) |
| `viewer/shapes_styling.py` | 1 (unnamed index), 3 (alpha-0 hide), 4 (expression layer/smoother), 8 (`source_row_hidden_mask`) |
| `viewer/adapter.py` | 7 (loguru fix), 8 (geometry exclusion — `ensure_styled_shapes_loaded`, `_build_shapes_layer`, `_prepare_napari_shapes_layer_inputs`) |
| `viewer/labels_styling.py` | 3 (alpha-0 hide), 4 (expression layer/smoother) |
| `viewer/_styling.py` | 4 (expression resolution) |
| `core/_color_source.py` | 4 (layer/smoother fields) |
| `tests/test_viewer_widget.py` | 3, 4, 5 |

To PR one of features 2–5 in isolation you would need to hand-split these files by hunk. Practical
options: (a) PR the self-contained features (6, then 3, then 7) first; (b) PR features 4+5 together
(both are the "colour source" UX) and 1+2 as small standalone fixes; (c) squash 2–5 into one
"viewer-card UX" PR. The per-function symbol lists above are the split boundaries.
