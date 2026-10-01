"""
utils.py
---------
Small helper functions used by the chatbot for:
  - detecting a laptop brand mentioned in free text
  - detecting a damage type mentioned in free text
  - detecting a purchase date
  - looking up brand-specific support info

These are lightweight regex/dictionary helpers that sit *around* the ML
model: the model decides *what the user wants* (the intent), these helpers
extract *which laptop and details* (the slots/entities).
"""

import re

BRAND_PATTERN = re.compile(
    r"\b(dell|hp|hewlett\s*packard|lenovo|asus|acer|apple|macbook|msi|samsung|toshiba|sony\s*vaio|microsoft\s*surface)\b",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"
)

BRAND_INFO = {
    "dell": {"name": "Dell", "id_label": "Service Tag", "site": "Dell Support (support.dell.com)"},
    "hp": {"name": "HP", "id_label": "Serial Number", "site": "HP Support (support.hp.com)"},
    "lenovo": {"name": "Lenovo", "id_label": "Serial Number", "site": "Lenovo Support (support.lenovo.com)"},
    "asus": {"name": "Asus", "id_label": "Serial Number", "site": "Asus Support (www.asus.com/support)"},
    "acer": {"name": "Acer", "id_label": "Serial Number", "site": "Acer Support (www.acer.com/support)"},
    "apple": {"name": "Apple", "id_label": "Serial Number", "site": "Apple Support (checkcoverage.apple.com)"},
    "macbook": {"name": "Apple", "id_label": "Serial Number", "site": "Apple Support (checkcoverage.apple.com)"},
    "msi": {"name": "MSI", "id_label": "Serial Number", "site": "MSI Support (www.msi.com/support)"},
    "samsung": {"name": "Samsung", "id_label": "Serial Number", "site": "Samsung Support (www.samsung.com/support)"},
}

DAMAGE_KEYWORDS = [
    (r"crack|screen", "a cracked/damaged screen"),
    (r"spill|liquid|water", "liquid spill damage"),
    (r"fell|drop|dent|shatter", "physical drop damage"),
    (r"keyboard", "keyboard damage"),
    (r"battery", "a battery issue"),
]


def detect_brand(text: str):
    m = BRAND_PATTERN.search(text)
    return m.group(0) if m else None


def detect_date(text: str):
    m = DATE_PATTERN.search(text)
    return m.group(0) if m else None


def detect_damage_type(text: str):
    t = text.lower()
    for pattern, label in DAMAGE_KEYWORDS:
        if re.search(pattern, t):
            return label
    return None


def detect_yes_no(text: str):
    t = text.lower()
    if re.search(r"\b(yes|yeah|yep|have it|i do)\b", t):
        return True
    if re.search(r"\b(no|nope|don't|dont|not sure|i do not)\b", t):
        return False
    return None


def brand_info(brand: str):
    if not brand:
        return None
    return BRAND_INFO.get(brand.lower().strip(), {
        "name": brand.strip().title(),
        "id_label": "Serial Number",
        "site": "your laptop brand's official support site",
    })
