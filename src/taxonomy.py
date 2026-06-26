"""Unified semantic taxonomy shared across datasets.

Mixing datasets safely (the harmonization step) requires every source to emit the
SAME small label set, otherwise the model wastes capacity reconciling synonyms
("Hall" vs "Corridor", "LivingRoom" vs "Livingroom"). We collapse everything to
~14 unified labels and map each dataset's raw classes onto them.
"""

# Final label vocabulary the model is asked to produce.
UNIFIED_LABELS = [
    "bedroom", "living_room", "kitchen", "dining", "bathroom", "corridor",
    "balcony", "storage", "stairwell", "office", "garage", "outdoor", "utility", "room",
]

# ── CubiCasa5K raw room class -> unified label ────────────────────────────────
# (Same coverage as the original parser, but targets the unified set; note
#  CubiCasa "Hall/Entry/..." now map to "corridor" to align with MSD.)
CUBICASA_ROOM_MAP = {
    "Alcove": "room", "Attic": "room", "Ballroom": "room", "Bar": "room", "Basement": "room",
    "Bath": "bathroom", "Bedroom": "bedroom", "Below150cm": "room", "CarPort": "garage",
    "Church": "room", "Closet": "storage", "ConferenceRoom": "room", "Conservatory": "room",
    "Counter": "room", "Den": "room", "Dining": "dining", "DraughtLobby": "corridor",
    "DressingRoom": "storage", "EatingArea": "dining", "Elevated": "room", "Elevator": "room",
    "Entry": "corridor", "ExerciseRoom": "room", "Garage": "garage", "Garbage": "room",
    "Hall": "corridor", "HallWay": "corridor", "HotTub": "room", "Kitchen": "kitchen",
    "Library": "room", "LivingRoom": "living_room", "Loft": "room", "Lounge": "living_room",
    "MediaRoom": "room", "MeetingRoom": "room", "Museum": "room", "Nook": "room",
    "Office": "office", "OpenToBelow": "room", "Outdoor": "outdoor", "Pantry": "room",
    "Reception": "room", "RecreationRoom": "room", "RetailSpace": "room", "Room": "room",
    "Sanctuary": "room", "Sauna": "bathroom", "ServiceRoom": "room", "ServingArea": "room",
    "Skylights": "room", "Stable": "room", "Stage": "room", "StairWell": "stairwell",
    "Storage": "storage", "SunRoom": "room", "SwimmingPool": "room", "TechnicalRoom": "room",
    "Theatre": "room", "Undefined": "room", "UserDefined": "room", "Utility": "utility",
}

# ── MSD (Modified Swiss Dwellings) ────────────────────────────────────────────
# Integer pixel values in full_out/*.npy follow MSD's ROOM_NAMES order (see the
# MSD repo constants.py): 0..8 rooms, 9 Structure(walls), 10 Door, 11 Entrance
# Door, 12 Window.  VERIFY against a real array with:  python -m src.data_msd <file.npy>
MSD_ROOM_INDICES = {
    0: "bedroom",      # Bedroom
    1: "living_room",  # Livingroom
    2: "kitchen",      # Kitchen
    3: "dining",       # Dining
    4: "corridor",     # Corridor
    5: "stairwell",    # Stairs
    6: "storage",      # Storeroom
    7: "bathroom",     # Bathroom
    8: "balcony",      # Balcony
}
MSD_STRUCTURE_INDEX = 9
MSD_DOOR_INDICES = (10, 11)   # Door, Entrance Door
MSD_WINDOW_INDEX = 12

# ── Structured3D ──────────────────────────────────────────────────────────────
# semantic 'type' strings from the official misc/colors.py -> unified label.
# ('door'/'window'/'outwall' are handled separately, not as rooms.)
S3D_ROOM_MAP = {
    "living room": "living_room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "balcony": "balcony",
    "corridor": "corridor",
    "dining room": "dining",
    "study": "office",
    "studio": "room",
    "store room": "storage",
    "garden": "outdoor",
    "laundry room": "utility",
    "office": "office",
    "basement": "room",
    "garage": "garage",
    "undefined": "room",
}
