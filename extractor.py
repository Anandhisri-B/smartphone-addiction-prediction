"""
extractor.py  (v6 - Multi-Brand Android + Stock Android Support)
=================================================================
Supports:
  1. Samsung Digital Wellbeing  → Category-level pages (Social, Dashboard, Video...)
  2. Stock Android / Nokia / OnePlus / Realme / Oppo / Vivo / Pixel
       → "App activity details" dashboard with app list
       → Individual app detail pages (AppName + "X minutes" / "X hrs, Y mins")

Brand is auto-detected from OCR text — no manual input needed.

Nokia/Stock Android screenshot types:
  TYPE A: App activity details (LIST PAGE)  ← MOST IMPORTANT
    Nav bar: "App activity details"
    Big time: "6 hrs, 22 mins"
    App rows: "YouTube" then "2 hrs, 18 mins" (two-line format)

  TYPE B: Individual app detail page
    App name big: "Chrome" / "WhatsApp" / "Instagram"
    Buttons: "Screen time ▼  Hourly ▼"
    Big time: "26 minutes" / "1 hr, 21 mins"
    Date: "Mon, 2 Feb"

Duplicate removal (two layers):
  1. Text hash   — exact same OCR content
  2. pHash       — visually near-identical images (hamming distance <= 10)

Which screenshots to upload:
  Nokia/Stock Android:
    MUST HAVE:  App activity details main page (gives total + all top apps)
    OPTIONAL:   Individual app pages for apps NOT visible in main list
    NOT NEEDED: Duplicate main pages, weekly if daily exists
  Samsung:
    MUST HAVE:  Dashboard page (total + notifications + unlocks)
    MUST HAVE:  Social / Productivity / Video category pages
    NOT NEEDED: Duplicate pages, same category daily + weekly (daily preferred)
"""

import re
import os
import cv2
import numpy as np
import pytesseract
from pathlib import Path


# ── Windows: Auto-set Tesseract path ─────────────────────────────────────────
def _set_tesseract_path():
    import platform
    if platform.system() != 'Windows':
        return
    candidates = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.join(os.environ.get('LOCALAPPDATA', ''),
                     r'Programs\Tesseract-OCR\tesseract.exe'),
        os.path.join(os.environ.get('USERPROFILE', ''),
                     r'AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
    ]
    for path in candidates:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

_set_tesseract_path()


# ═══════════════════════════════════════════════════════════════════════════════
#  APP → CATEGORY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
APP_CATEGORY = {
    # Social / messaging
    'whatsapp': 'social', 'instagram': 'social', 'facebook': 'social',
    'messenger': 'social', 'telegram': 'social', 'snapchat': 'social',
    'twitter': 'social', 'x': 'social', 'tiktok': 'social',
    'linkedin': 'social', 'discord': 'social', 'signal': 'social',
    'wechat': 'social', 'viber': 'social', 'hike': 'social',
    'sharechat': 'social', 'moj': 'social', 'josh': 'social',
    'phone': 'social', 'truecaller': 'social', 'contacts': 'social',
    'dialer': 'social', 'messages': 'social', 'sms': 'social',
    'kik': 'social', 'skype': 'social', 'hangouts': 'social',

    # Video / streaming
    'youtube': 'video', 'netflix': 'video', 'hotstar': 'video',
    'disney': 'video', 'amazon prime': 'video', 'prime video': 'video',
    'mx player': 'video', 'vlc': 'video', 'zee5': 'video',
    'sony liv': 'video', 'sonyliv': 'video', 'voot': 'video',
    'jio cinema': 'video', 'jiocinema': 'video', 'mxplayer': 'video',
    'dailymotion': 'video', 'twitch': 'video',

    # Gaming
    'pubg': 'gaming', 'bgmi': 'gaming', 'free fire': 'gaming',
    'freefire': 'gaming', 'clash of clans': 'gaming', 'clash royale': 'gaming',
    'candy crush': 'gaming', 'ludo king': 'gaming', 'among us': 'gaming',
    'minecraft': 'gaming', 'roblox': 'gaming', 'mobile legends': 'gaming',
    'pokemon': 'gaming', 'game': 'gaming', 'games': 'gaming',
    'play games': 'gaming',

    # Productivity / tools
    'chrome': 'productivity', 'google': 'productivity', 'gmail': 'productivity',
    'drive': 'productivity', 'docs': 'productivity', 'sheets': 'productivity',
    'slides': 'productivity', 'maps': 'productivity', 'calendar': 'productivity',
    'keep': 'productivity', 'tasks': 'productivity', 'chatgpt': 'productivity',
    'copilot': 'productivity', 'edge': 'productivity', 'firefox': 'productivity',
    'brave': 'productivity', 'opera': 'productivity', 'calculator': 'productivity',
    'clock': 'productivity', 'settings': 'productivity', 'files': 'productivity',
    'gallery': 'productivity', 'photos': 'productivity', 'camera': 'productivity',
    'notes': 'productivity', 'microsoft': 'productivity', 'outlook': 'productivity',
    'teams': 'productivity', 'zoom': 'productivity', 'meet': 'productivity',
    'where is my train': 'productivity', 'train': 'productivity',
    'paytm': 'productivity', 'gpay': 'productivity', 'phonepe': 'productivity',
    'amazon': 'productivity', 'flipkart': 'productivity',
    'swiggy': 'productivity', 'zomato': 'productivity',
    'ola': 'productivity', 'uber': 'productivity',
}

# Samsung nav bar → feature category
SAMSUNG_CATEGORY_MAP = {
    'social': 'social',
    'productivity and finance': 'productivity',
    'productivity': 'productivity',
    'video': 'video',
    'gaming': 'gaming',
    'dashboard': 'dashboard',
    'entertainment': 'video',
    'communication': 'social',
    'utilities': 'productivity',
    'travel': 'productivity',
    'health and fitness': 'others',
    'health': 'others',
    'shopping': 'others',
    'food and drink': 'others',
    'music and audio': 'others',
    'news and magazines': 'others',
    'books and reference': 'others',
    'education': 'productivity',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def parse_mins(s):
    """
    Convert any time string to total minutes.
    Handles all formats including OCR noise around values:
      Stock Android: '26 minutes', '1 hr, 21 mins', '2 hrs, 18 mins', '1 minute'
                     'SS 31 minutes @O'  (OCR icon noise around value)
                     '6 1 minute g'      (OCR icon prefix)
                     '* 13 minutes'      (bullet OCR noise)
      Samsung:       '2h 29m', '5h14m', '18m', '1h 6m / day'
    """
    s = str(s).strip().lower()

    # "X hrs, Y mins" / "X hr, Y min" (Stock Android)
    m = re.search(r'(\d+)\s*hr[s]?\s*[,]?\s*(\d+)\s*min', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    # "X hrs" / "X hr" only
    m = re.search(r'(\d+)\s*hr[s]?\b', s)
    if m:
        return int(m.group(1)) * 60

    # "X minutes" / "X mins" / "X minute" — robust: allows noise around the number
    m = re.search(r'\b(\d+)\s*min(?:ute[s]?)?(?!\w)', s)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 720:
            return v

    # Samsung compact: "2h 29m", "5h14m"
    m = re.search(r'(\d+)\s*h\s*(\d+)\s*m', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    m = re.search(r'(\d+)\s*h\b', s)
    if m:
        return int(m.group(1)) * 60

    m = re.search(r'(\d+)\s*m(?![a-z])', s)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 720:
            return v

    return 0


def ocr_image(path):
    """Preprocess and OCR a single image."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if gray2.mean() < 128:
        gray2 = cv2.bitwise_not(gray2)
    _, thresh = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(thresh, config='--psm 6 --oem 3')


def phash(img_path, hash_size=8):
    """Perceptual hash for near-duplicate visual detection."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size * 4, hash_size * 4))
    dct_result = cv2.dct(np.float32(resized))
    dct_low = dct_result[:hash_size, :hash_size]
    med = np.median(dct_low)
    return ''.join(['1' if v > med else '0' for v in dct_low.flatten()])


def hamming(h1, h2):
    """Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def app_to_category(app_name):
    """Map an app name to a feature category."""
    name = app_name.lower().strip()
    if name in APP_CATEGORY:
        return APP_CATEGORY[name]
    for key, cat in APP_CATEGORY.items():
        if key in name or name in key:
            return cat
    return 'others'


def detect_daily(lines):
    """Return True if screenshot shows a daily (not weekly) view."""
    for line in lines[:8]:
        ll = line.lower()
        # Daily indicator: specific weekday name visible
        if re.search(r'\b(mon|tue|wed|thu|fri|sat|sun)\b', ll):
            # But NOT the weekly bar chart label row (has multiple days)
            day_count = len(re.findall(r'\b(mon|tue|wed|thu|fri|sat|sun)\b', ll))
            if day_count == 1:
                return True
        if 'weekly view' in ll:
            return True   # "Weekly View" button = currently on daily page
        if 'daily view' in ll:
            return False  # "Daily View" button = currently on weekly page
    return True  # default: assume daily


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAND + PAGE TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_ocr(text):
    """
    Fix common OCR merging errors from Stock Android screenshots:
      'Appactivity' → 'app activity'
      'Screentime'  → 'screen time'
      'AppActivity' → 'app activity'
    """
    t = text.lower()
    t = re.sub(r'appactivity', 'app activity', t)
    t = re.sub(r'screentime', 'screen time', t)
    t = re.sub(r'app\s+activity\s+details', 'app activity details', t)
    return t


def detect_brand(lines):
    """
    Auto-detect phone brand and page type.

    Returns: (brand, page_type)
      brand:     'samsung' | 'stock_android'
      page_type: 'samsung_dashboard' | 'samsung_category' |
                 'stock_list' | 'stock_app_detail' | 'unknown'

    Stock Android covers: Nokia, OnePlus, Realme, Oppo, Vivo, Pixel,
                          Motorola, Xiaomi (with stock-like UI)
    """
    full_text = _normalize_ocr(' '.join(lines[:8]))

    # ── Stock Android markers ────────────────────────────────────────────────
    # Main list page: nav bar says "App activity details" (may be OCR-merged)
    if 'app activity details' in full_text or 'appactivity' in ' '.join(lines[:3]).lower():
        return 'stock_android', 'stock_list'

    # Individual app page: has "Screen time" + "Hourly" pill buttons
    if ('screen time' in full_text or 'screentime' in full_text) and 'hourly' in full_text:
        return 'stock_android', 'stock_app_detail'

    # ── Samsung markers ──────────────────────────────────────────────────────
    for line in lines[:5]:
        ll = re.sub(r'^[<\s€]+', '', line.lower().strip()).strip()
        if ll == 'dashboard':
            return 'samsung', 'samsung_dashboard'
        for key in SAMSUNG_CATEGORY_MAP:
            if ll == key or ll.startswith(key):
                return 'samsung', 'samsung_category'

    return 'unknown', 'unknown'


# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK ANDROID PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

# Lines to skip when parsing app lists
_STOCK_SKIP = re.compile(
    r'^(\d+h|[<>]|screen\s?time|hourly|app\s+activity|show\s+sites|'
    r'sun\s+mon|mon\s+tue|\d+:\d+|settings|app\s+timer|no\s+timer|'
    r'manage\s+notif|app\s+used\s+in)',
    re.IGNORECASE
)


def _clean_app_name(name):
    """
    Strip OCR icon/noise characters from start/end of app names.
    Examples:
      'gy youtube 2'  → 'youtube'
      '© Chrome R'    → 'chrome'
      '& 31 minutes'  → '' (invalid — is a time line)
      '-») Photos'    → 'photos'
      'SS 31 minutes' → '' (skip — this is a time line with noise)
    """
    # Remove trailing OCR noise (single chars, icons, digits)
    name = re.sub(r'[\s\d©@\.\•\*\-\»\(\)R\|l\[\]]{0,3}\s*$', '', name).strip()
    # Remove leading OCR noise (non-letter chars and short noise tokens)
    name = re.sub(r'^[^a-zA-Z]+', '', name).strip()
    # Remove short noise tokens at start (2 chars or fewer before real name)
    name = re.sub(r'^[a-z]{1,2}\s+(?=[a-zA-Z])', '', name).strip()
    return name.lower()


def parse_stock_list_page(lines, is_daily):
    """
    Parse "App activity details" main list page (Stock Android).

    Returns: {'total_min': int, 'apps': {app_name_lower: minutes}}

    Two-line format (most common):
      "YouTube"
      "2 hrs, 18 mins"

    One-line format (OCR sometimes merges):
      "YouTube  2 hrs, 18 mins"
    """
    result = {'total_min': 0, 'apps': {}}

    # ── Total screen time ─────────────────────────────────────────────────
    for line in lines[1:7]:
        ll = line.lower().strip()
        if _STOCK_SKIP.match(ll):
            continue
        if re.match(r'^(mon|tue|wed|thu|fri|sat|sun),?\s+\d+', ll):
            continue
        mins = parse_mins(ll)
        if mins >= 30:
            result['total_min'] = mins
            break

    # ── Find where app list starts (after weekly bar chart) ───────────────
    list_start = 0
    for i, line in enumerate(lines):
        ll = line.lower()
        # Row with multiple day abbreviations = bar chart label row
        if re.search(r'\bsun\b.*\bmon\b|\bmon\b.*\btue\b', ll):
            list_start = i + 2   # skip chart + nav row
            break
        # Navigation row like "< Mon, 2 Feb >"
        if re.match(r'^<\s+(mon|tue|wed|thu|fri|sat|sun)', ll):
            list_start = i + 1
            break

    app_lines = [l.strip() for l in lines[list_start:]
                 if l.strip() and not _STOCK_SKIP.match(l.strip().lower())]

    # ── Parse app entries ─────────────────────────────────────────────────
    last_app_name = None  # track last seen app name for orphan time lines

    i = 0
    while i < len(app_lines):
        line = app_lines[i]
        ll = line.lower()

        # Skip date lines like "Mon, 2 Feb"
        if re.match(r'^(mon|tue|wed|thu|fri|sat|sun),?\s+\d+', ll):
            i += 1
            continue

        # Skip "Show sites that you visit" button
        if 'show sites' in ll:
            i += 1
            continue

        # One-line format: "AppName   X hrs, Y mins"
        one = re.match(
            r'^(.+?)\s{2,}(\d+\s*hr[s]?,?\s*\d+\s*min[s]?|\d+\s*hr[s]?'
            r'|\d+\s*min(?:ute[s]?)?)\s*$',
            line, re.IGNORECASE
        )
        if one:
            app_name = _clean_app_name(one.group(1))
            mins = parse_mins(one.group(2))
            if mins > 0 and len(app_name) >= 2:
                result['apps'][app_name] = mins
                last_app_name = app_name
            i += 1
            continue

        # Check if current line is a time line (possibly with OCR noise)
        cur_mins = parse_mins(ll)
        if cur_mins > 0 and re.search(r'\bmin(?:ute[s]?)?|\bhr[s]?\b', ll):
            # Standalone time line — belongs to last seen app name
            if last_app_name:
                result['apps'][last_app_name] = cur_mins
            i += 1
            continue

        # Two-line format: app name on this line, time on next line
        if i + 1 < len(app_lines):
            next_line = app_lines[i + 1]
            next_mins = parse_mins(next_line.lower())
            if next_mins > 0 and re.search(r'\bmin(?:ute[s]?)?|\bhr[s]?\b', next_line.lower()):
                app_name = _clean_app_name(line)
                if re.search(r'[a-zA-Z]', app_name) and len(app_name) >= 2:
                    result['apps'][app_name] = next_mins
                    last_app_name = app_name
                i += 2
                continue

        # This line is an app name without a parseable time following it
        potential_name = _clean_app_name(line)
        if re.search(r'[a-zA-Z]{2,}', potential_name):
            last_app_name = potential_name

        i += 1

    return result


def parse_stock_app_detail(lines, is_daily):
    """
    Parse individual app detail page (Stock Android).
    Returns: (app_name_str, minutes_int)

    Layout:
      ← [back arrow]
      AppName          ← big heading
      Screen time ▼  Hourly ▼
      26 minutes       ← usage
      Mon, 2 Feb
    """
    app_name = None
    minutes = 0

    # App name: second or third non-empty non-system line
    for line in lines[:8]:
        ll = line.lower().strip()
        if any(kw in ll for kw in ['screen time', 'hourly', 'app activity',
                                    'settings', 'app timer', 'manage']):
            continue
        if re.match(r'^\d+:\d+', ll):  # clock time
            continue
        if re.match(r'^[€<\s]+$', ll):  # back arrow only
            continue
        if re.search(r'[a-zA-Z]{2,}', line) and not re.match(r'^\d+', line.strip()):
            app_name = re.sub(r'^[^a-zA-Z]+', '', line).strip()
            if app_name:
                break

    # Usage time: first valid time in lines 2-9
    for line in lines[2:10]:
        ll = line.lower().strip()
        if any(kw in ll for kw in ['screen time', 'hourly', 'settings',
                                    'app timer', 'manage', 'background']):
            continue
        if re.match(r'^(mon|tue|wed|thu|fri|sat|sun)', ll):
            continue
        mins = parse_mins(ll)
        if mins > 0:
            minutes = mins
            break

    return app_name, minutes


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMSUNG PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def samsung_detect_category(lines):
    for line in lines[:5]:
        ll = re.sub(r'^[<\s€]+', '', line.lower().strip()).strip()
        for key, cat in SAMSUNG_CATEGORY_MAP.items():
            if ll == key or ll.startswith(key):
                return cat
    return 'unknown'


def samsung_parse_category_time(lines, is_daily):
    for line in lines[3:12]:
        ll = line.lower()
        if any(kw in ll for kw in ['app timer', 'view all', 'notification',
                                    'app open', 'unlock', 'weekly', 'daily',
                                    'dashboard', 'show app']):
            continue
        clean = re.sub(r'[^0-9hHmM\s]', '', line).strip()
        if is_daily:
            mins = parse_mins(clean)
            if mins >= 5:
                return mins
        else:
            m = re.search(r'(\d+\s*h\s*\d+\s*m|\d+\s*m)\s*/\s*day', ll)
            if m:
                return parse_mins(m.group(1))
    return 0


def samsung_parse_notifications_unlocks(lines, is_daily):
    notifs, unlocks = -1, -1
    for line in lines:
        ll = line.lower().strip()
        m = re.match(r'^(\d+)\s+notification', ll)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 2000:
                notifs = max(notifs, v)
            continue
        m = re.match(r'^(\d+)\s+unlock', ll)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 1000:
                unlocks = v
    return max(notifs, 0), max(unlocks, 0)


def samsung_parse_dashboard(lines, is_daily):
    result = {'total_min': 0, 'social': 0, 'productivity': 0,
              'video': 0, 'gaming': 0, 'others': 0,
              'notifications': -1, 'unlocks': -1}
    for line in lines[3:10]:
        ll = line.lower()
        if any(kw in ll for kw in ['notification', 'unlock', 'social',
                                    'productivity', 'video', 'dashboard']):
            continue
        clean = re.sub(r'[^0-9hHmM\s]', '', line).strip()
        mins = parse_mins(clean)
        if is_daily and mins >= 10:
            result['total_min'] = mins
            break
    CAT_ROW = re.compile(
        r'[@\.\•\*\s]*(social|productivity and finance|productivity|video|gaming)'
        r'\s+(\d+\s*h\s*\d+\s*m|\d+\s*h|\d+\s*m)',
        re.IGNORECASE
    )
    for line in lines:
        m = CAT_ROW.search(line.lower())
        if m:
            cat_name = m.group(1).strip()
            mins = parse_mins(m.group(2))
            mapped = SAMSUNG_CATEGORY_MAP.get(cat_name, 'others')
            if mapped in result and mins > 0:
                result[mapped] = max(result.get(mapped, 0), mins)
    n, u = samsung_parse_notifications_unlocks(lines, is_daily)
    result['notifications'] = n
    result['unlocks'] = u
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DUPLICATE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DuplicateDetector:
    """
    Two-layer duplicate detection:
      Layer 1: Text hash  — identical OCR content
      Layer 2: pHash      — visually near-identical (hamming distance <= threshold)
    """
    def __init__(self, phash_threshold=10):
        self.text_hashes = set()
        self.phashes = []          # list of (hash_str, img_name)
        self.threshold = phash_threshold

    def is_duplicate(self, text, img_path):
        # Layer 1: text content hash
        th = hash(text[:400].strip())
        if th in self.text_hashes:
            return True, 'text_duplicate'
        self.text_hashes.add(th)

        # Layer 2: perceptual image hash
        ph = phash(img_path)
        if ph:
            for existing_ph, existing_name in self.phashes:
                dist = hamming(ph, existing_ph)
                if dist <= self.threshold:
                    return True, f'visual_duplicate_of_{existing_name}'
            self.phashes.append((ph, Path(img_path).name))

        return False, None


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_user_data(user_folder):
    """
    Process all screenshots in a folder → 18-feature dict.

    Auto-detects brand and page type per screenshot.
    Removes duplicates automatically (text + visual hash).
    Merges data: daily beats weekly, list page beats individual app pages.
    """
    folder = Path(user_folder)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

    if not images:
        raise ValueError(f"No images found in {folder}")

    # ── Accumulators ─────────────────────────────────────────────────────────
    daily  = {'total': 0, 'social': 0, 'productivity': 0, 'video': 0,
              'gaming': 0, 'others': 0, 'notifications': -1, 'unlocks': -1}
    weekly = {k: v for k, v in daily.items()}

    # For Stock Android: track if we have the authoritative list page
    list_page_seen = False
    list_page_app_names = set()   # app names already captured from list page

    dup = DuplicateDetector(phash_threshold=10)

    for img_path in images:
        try:
            text = ocr_image(str(img_path))
        except Exception as e:
            print(f"    Skipping {img_path.name}: {e}")
            continue

        # ── Duplicate check ──────────────────────────────────────────────────
        is_dup, reason = dup.is_duplicate(text, str(img_path))
        if is_dup:
            print(f"    {img_path.name} → DUPLICATE ({reason}) — skipped")
            continue

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        brand, page_type = detect_brand(lines)
        is_daily = detect_daily(lines)
        target = daily if is_daily else weekly

        print(f"    {img_path.name} → brand={brand}  page={page_type}  daily={is_daily}")

        # ════════════════════════════════════════════════════════════════════
        #  STOCK ANDROID
        # ════════════════════════════════════════════════════════════════════
        if brand == 'stock_android':

            if page_type == 'stock_list':
                parsed = parse_stock_list_page(lines, is_daily)

                # Store total
                if parsed['total_min'] > 0:
                    target['total'] = max(target['total'], parsed['total_min'])
                    if is_daily:
                        list_page_seen = True

                # Accumulate per-category from app list
                cat_totals = {}
                for app_name, mins in parsed['apps'].items():
                    cat = app_to_category(app_name)
                    cat_totals[cat] = cat_totals.get(cat, 0) + mins
                    if is_daily:
                        list_page_app_names.add(app_name)

                for cat, total in cat_totals.items():
                    if cat in target:
                        target[cat] = max(target[cat], total)

                print(f"      total={parsed['total_min']}  "
                      f"apps_found={len(parsed['apps'])}  "
                      f"sample={list(parsed['apps'].items())[:4]}")

            elif page_type == 'stock_app_detail':
                app_name, mins = parse_stock_app_detail(lines, is_daily)
                if app_name and mins > 0:
                    app_lower = app_name.lower().strip()
                    # Skip if this app was already captured from the list page
                    if list_page_seen and app_lower in list_page_app_names:
                        print(f"      {app_name} ({mins}m) — already in list page, skipped")
                    else:
                        cat = app_to_category(app_name)
                        target[cat] = max(target[cat], mins)
                        print(f"      app={app_name}  mins={mins}  cat={cat}")

        # ════════════════════════════════════════════════════════════════════
        #  SAMSUNG
        # ════════════════════════════════════════════════════════════════════
        elif brand == 'samsung':

            if page_type == 'samsung_dashboard':
                dash = samsung_parse_dashboard(lines, is_daily)
                if is_daily:
                    if dash['total_min'] > 0:
                        daily['total'] = max(daily['total'], dash['total_min'])
                    for cat in ('social', 'productivity', 'video', 'gaming', 'others'):
                        if dash[cat] > 0:
                            daily[cat] = max(daily[cat], dash[cat])
                    if dash['notifications'] > 0:
                        daily['notifications'] = max(daily['notifications'], dash['notifications'])
                    if dash['unlocks'] >= 0:
                        daily['unlocks'] = max(daily['unlocks'], dash['unlocks'])

            elif page_type == 'samsung_category':
                cat = samsung_detect_category(lines)
                cat_time = samsung_parse_category_time(lines, is_daily)
                if cat_time > 0:
                    target[cat] = max(target[cat], cat_time)
                n, _ = samsung_parse_notifications_unlocks(lines, is_daily)
                if n > 0:
                    target['notifications'] = max(target['notifications'], n)

        else:
            print(f"      Unknown brand/page — attempting generic extraction")
            for line in lines[2:6]:
                mins = parse_mins(line.lower())
                if mins >= 20:
                    target['total'] = max(target['total'], mins)
                    break

    # ── Merge daily + weekly (daily wins) ─────────────────────────────────────
    def pick(key):
        dv = daily.get(key) or 0
        wv = weekly.get(key) or 0
        return dv if dv > 0 else wv

    def pick_meta(key):
        dv = daily.get(key, -1)
        wv = weekly.get(key, -1)
        if dv is not None and dv >= 0:
            return dv
        if wv is not None and wv >= 0:
            return wv
        return 0

    social        = pick('social')
    productivity  = pick('productivity')
    video         = pick('video')
    gaming        = pick('gaming')
    others        = pick('others')
    total_notifs  = pick_meta('notifications')
    total_unlocks = pick_meta('unlocks')

    total = daily.get('total') or 0
    if total == 0:
        total = weekly.get('total') or 0
    if total == 0:
        total = social + productivity + video + gaming + others
    total = max(total, 1)

    total_hours  = max(total / 60, 0.1)
    prod_safe    = max(productivity, 1)
    sprod        = round(social / prod_safe, 2)
    night_min    = round(social * 0.12 + video * 0.10)

    print(f"\n    FINAL → total={total}  social={social}  prod={productivity}  "
          f"video={video}  gaming={gaming}  notifs={total_notifs}  unlocks={total_unlocks}")

    return {
        'total_screen_time_min':     total,
        'social_time_min':           social,
        'productivity_time_min':     productivity,
        'video_time_min':            video,
        'gaming_time_min':           gaming,
        'others_time_min':           others,
        'night_10pm_2am_min':        night_min,
        'early_morning_2am_6am_min': round(night_min * 0.3),
        'morning_6am_12pm_min':      round(total * 0.18),
        'afternoon_12pm_6pm_min':    round(total * 0.45),
        'evening_6pm_10pm_min':      round(total * 0.25),
        'total_notifications':       total_notifs,
        'total_unlocks':             total_unlocks,
        'social_prod_ratio':         min(sprod, 9.99),
        'night_usage_pct':           round(night_min / total * 100, 2),
        'gaming_pct':                round(gaming   / total * 100, 2),
        'notification_per_hour':     round(total_notifs  / total_hours, 2),
        'unlock_frequency_per_hour': round(total_unlocks / total_hours, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    result = extract_user_data(folder)
    print("\n── Extracted Features ──")
    for k, v in result.items():
        print(f"  {k}: {v}")