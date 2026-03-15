from . import aura, firecracker, neon, lightning, bubbles, grid_shadow, animal, matrix_human, infrared, kinetic_brush, flora_infusion, energy_master, gravity_pull, magic_spells, portal

# Registry: display name → module with apply(canvas, pose) function
FILTER_REGISTRY = {
    "Default":      None,          # plain skeleton, no extra filter
    "Magic Spells": magic_spells,
    "Gravity Pull": gravity_pull,
    "Portal":       portal,
    "Energy Master":energy_master,
    "Aura":         aura,
    "Neon":        neon,
    "Lightning":   lightning,
    "Firecracker": firecracker,
    "Bubbles":     bubbles,
    "Grid Shadow": grid_shadow,
    "Animal":      animal,
    "Kinetic Brush": kinetic_brush,
    "Flora Infusion": flora_infusion,
    "Matrix Human": matrix_human,
    "Infrared":    infrared,
}

FILTER_NAMES = list(FILTER_REGISTRY.keys())
