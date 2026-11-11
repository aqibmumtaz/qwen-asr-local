from datetime import date

from app.nlp import normalizers as n


def test_words_to_int():
    assert n.words_to_int("paanch sau") == 500
    assert n.words_to_int("dedh hazaar") == 1500
    assert n.words_to_int("dhai sau") == 250
    assert n.words_to_int("painteesh") == 35
    assert n.words_to_int("ek hazaar chaar sau pachaas") == 1450
    assert n.words_to_int("1450") == 1450
    assert n.words_to_int("hello") is None


def test_age_and_house_numbers():
    assert n.extract_age("umar painteesh saal") == 35
    assert n.extract_age("meri umar 42 hai") == 42
    assert 45 in n.find_numbers("ghar number paintaalees")


def test_phone_digits_and_words():
    assert n.normalize_phone("0300 1234567").digits == "03001234567"
    assert n.normalize_phone("0300 1234567").valid
    assert n.normalize_phone("zero teen zero zero ek do teen char paanch chay saat").digits == "03001234567"
    assert n.normalize_phone("zero teen double zero ek do teen char paanch chay saat").digits == "03001234567"
    assert n.normalize_phone("+92 300 1234567").digits == "03001234567"
    assert not n.normalize_phone("0300 123").valid


def test_yes_no():
    assert n.yes_no("haan kar do") == "yes"
    assert n.yes_no("ji theek hai") == "yes"
    assert n.yes_no("nahi galat hai") == "no"
    assert n.yes_no("gulshan block 13") is None


def test_gender():
    assert n.extract_gender("woh larki hai") == "female"
    assert n.extract_gender("mard hain") == "male"
    assert n.extract_gender("kuch aur") is None


def test_name():
    assert n.normalize_name("mera naam muhammad ali hai") == "Muhammad Ali"
    assert n.normalize_name("muhammad ali, umar painteesh saal") == "Muhammad Ali"


def test_dates():
    today = date(2026, 10, 21)  # Wednesday
    s = n.resolve_date("kal subah", today)
    assert s.day == date(2026, 10, 22) and s.part == "morning"
    s = n.resolve_date("parson dopahar", today)
    assert s.day == date(2026, 10, 23) and s.part == "afternoon"
    s = n.resolve_date("agle somwar shaam", today)
    assert s.day == date(2026, 10, 26) and s.part == "evening"
    s = n.resolve_date("15 november subah 10 baje", today)
    assert s.day == date(2026, 11, 15) and s.hour == 10
    s = n.resolve_date("kal shaam 5 baje", today)
    assert s.hour == 17 and s.complete


def test_speak():
    assert n.speak_digits("0300") == "zero teen zero zero"
    assert n.speak_number(500) == "paanch sau"
    assert n.speak_number(1450) == "ek hazaar char sau pachaas"


def test_clean_text():
    assert n.clean_text("Nhi, kl subah!") == "nahi kal subah"


def test_date_negation_and_merged_words():
    today = date(2026, 10, 21)
    s = n.resolve_date("nahi kal nahi parson shaam", today)
    assert s.day == date(2026, 10, 23) and s.part == "evening"
    s = n.resolve_date("parsonshaam", today)
    assert s.day == date(2026, 10, 23) and s.part == "evening"


def test_date_words_with_spelling_noise():
    today = date(2026, 10, 21)
    assert n.resolve_date("kal subeh", today).part == "morning"
    assert n.resolve_date("parson shaaam", today).part == "evening"
    assert n.resolve_date("kal dopehar", today).part == "afternoon"
    # names that look similar must not become date words
    assert n.resolve_date("subhan", today).part is None
    assert n.resolve_date("shamim", today).part is None
