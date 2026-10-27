from lpr.plate_text import PlateRules, normalize_text, parse, search_key, split_letters_digits, validate
from lpr.ocr import Char, assemble


def test_normalizes_variants_and_western_digits():
    assert normalize_text("أ ي ة 12") == "اىه١٢"
    assert normalize_text("ـبـ") == "ب"  # tatweel removed
    assert normalize_text("ABC-") == ""


def test_parse_valid_plate():
    plate = parse("ب ط ص ١٢٣٤")
    assert plate is not None
    assert plate.letters == "بطص"
    assert plate.digits == "١٢٣٤"
    assert plate.key == "بطص١٢٣٤"
    assert plate.display == "ب ط ص ١٢٣٤"


def test_parse_mixed_digit_scripts():
    assert parse("س ع 4 5 6").digits == "٤٥٦"


def test_rejects_bad_shapes():
    assert parse("ب ط ص ع ١٢٣") is None  # 4 letters
    assert parse("ب ١٢٣٤٥") is None  # 5 digits
    assert parse("١٢٣٤") is None  # no letters
    assert parse("ب ط ص") is None  # no digits
    assert parse("ب ١٢") is None  # too short
    assert parse("ب ط ٠١٢") is None  # leading zero


def test_custom_rules():
    rules = PlateRules(min_letters=2, max_letters=3, min_digits=3, max_digits=4, min_total=5)
    assert validate("ب", "١٢٣٤", rules) is None
    assert validate("بط", "١٢٣٤", rules) is not None


def test_split_and_search_key():
    assert split_letters_digits("ق ل 7 8 9") == ("قل", "٧٨٩")
    assert search_key(" ب ط 12 ") == "بط١٢"


def _c(text, x, y=20, w=20, h=30, conf=0.9):
    return Char(normalize_text(text), conf, x, y, x + w, y + h)


def test_assemble_orders_letters_rtl_and_digits_ltr():
    # EALPR sample 0001: image shows "٧ ١ ٢ ٦ | ن ط س" left to right.
    chars = [_c("٧", 10), _c("١", 32), _c("٢", 54), _c("٦", 76), _c("ن", 110), _c("ط", 135), _c("س", 160)]
    read = assemble(chars, PlateRules())
    assert read is not None
    assert read.plate.display == "س ط ن ٧١٢٦"


def test_assemble_ignores_band_and_small_noise():
    chars = [
        _c("م", 150, y=0, h=10), _c("ص", 165, y=0, h=10), _c("ر", 180, y=0, h=10),  # small band text
        _c("٥", 10), _c("٤", 32), _c("٣", 54),
        _c("ب", 120), _c("ا", 150, conf=0.5),
        _c("٩", 60, y=90),  # stray detection far below the row
    ]
    read = assemble(chars, PlateRules())
    assert read.plate.display == "ا ب ٥٤٣"
    assert 0.8 < read.conf < 0.9


def test_assemble_normalizes_dataset_class_names():
    # EALPR class names use "أ" and "ھ"
    read = assemble([_c("أ", 150), _c("ھ", 120), _c("١", 10), _c("٢", 32)], PlateRules())
    assert read.plate.key == "اه١٢"


def test_assemble_returns_none_without_plate():
    assert assemble([], PlateRules()) is None
    assert assemble([_c("ب", 120), _c("ط", 150)], PlateRules()) is None
