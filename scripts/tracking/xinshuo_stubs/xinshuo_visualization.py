"""Minimal stub for AB3DMOT's xinshuo_visualization dependency (only random_colors, used in
the vis path we don't call but which model.py imports at module load). ponytail: stub."""


def random_colors(N, bright=True):
    # deterministic evenly-spaced HSV->RGB wheel; we never actually render, so exact hues don't matter
    import colorsys
    hsv = [(i / max(N, 1), 1.0, 1.0 if bright else 0.7) for i in range(N)]
    return [colorsys.hsv_to_rgb(*c) for c in hsv]
