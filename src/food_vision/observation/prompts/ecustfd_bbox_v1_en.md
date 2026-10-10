You are shown a photo of a single whole fruit on a flat surface, taken
either from directly above or from the side. A 25 mm coin lies next to
the fruit as a scale reference.

Locate both objects precisely:

1. The coin — a small, round, flat object.
2. The fruit — the single piece of whole fruit in the photo.

For each, give a tight bounding box that touches the object's visible
edges on all four sides (not a loose box with extra margin). Report
coordinates as fractions of the image, where `(0, 0)` is the top-left
corner and `(1, 1)` is the bottom-right corner: `xmin`/`ymin` is the
box's top-left corner, `xmax`/`ymax` is its bottom-right corner, each a
number between 0 and 1.

First, in the `observations` field, briefly describe what you see: the
fruit type, roughly where the coin and the fruit are in the frame, and
the view (top or side). Then give the eight box coordinates:
`coin_xmin`, `coin_ymin`, `coin_xmax`, `coin_ymax`, `fruit_xmin`,
`fruit_ymin`, `fruit_xmax`, `fruit_ymax`.

Do not estimate mass, size in centimetres, calories or nutrition facts —
only the two bounding boxes are requested.
