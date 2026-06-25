"""Shared prompt + schema definitions (used identically by SFT and GRPO).

Keeping the prompt identical across SFT and GRPO matters: the RL stage must see the
same prompt distribution it was supervised on, otherwise the reward signal fights the
SFT prior. (The original reference scripts used a shorter GRPO system prompt — we fix
that here by sharing one definition.)
"""

SYSTEM_PROMPT = (
    "You are a floor plan vectorization expert. Extract wall, door, window geometry "
    "from floor plan images into structured JSON.\n\n"
    "Output ONLY valid JSON with this schema:\n"
    '{"walls":[{"id":"wall_N","start":[x,y],"end":[x,y],"thickness":T,"curvature":0,'
    '"openings":[{"type":"door"|"window","center":D,"width":W}]}],'
    '"rooms":[{"label":"room_type","walls":["wall_N",...]}]}\n\n'
    "Coordinates normalized so longer image edge = 1024."
)

USER_PROMPT = (
    "Vectorize this floor plan into structured JSON with all walls, doors, windows, and rooms."
)

# CubiCasa5K room class -> normalized label
ROOM_MAP = {
    "Alcove": "room", "Attic": "room", "Ballroom": "room", "Bar": "room", "Basement": "room",
    "Bath": "bathroom", "Bedroom": "bedroom", "Below150cm": "room", "CarPort": "garage",
    "Church": "room", "Closet": "storage", "ConferenceRoom": "room", "Conservatory": "room",
    "Counter": "room", "Den": "room", "Dining": "dining", "DraughtLobby": "hallway",
    "DressingRoom": "storage", "EatingArea": "dining", "Elevated": "room", "Elevator": "room",
    "Entry": "hallway", "ExerciseRoom": "room", "Garage": "garage", "Garbage": "room",
    "Hall": "hallway", "HallWay": "hallway", "HotTub": "room", "Kitchen": "kitchen",
    "Library": "room", "LivingRoom": "living_room", "Loft": "room", "Lounge": "living_room",
    "MediaRoom": "room", "MeetingRoom": "room", "Museum": "room", "Nook": "room",
    "Office": "office", "OpenToBelow": "room", "Outdoor": "outdoor", "Pantry": "room",
    "Reception": "room", "RecreationRoom": "room", "RetailSpace": "room", "Room": "room",
    "Sanctuary": "room", "Sauna": "bathroom", "ServiceRoom": "room", "ServingArea": "room",
    "Skylights": "room", "Stable": "room", "Stage": "room", "StairWell": "stairwell",
    "Storage": "storage", "SunRoom": "room", "SwimmingPool": "room", "TechnicalRoom": "room",
    "Theatre": "room", "Undefined": "room", "UserDefined": "room", "Utility": "utility",
}
