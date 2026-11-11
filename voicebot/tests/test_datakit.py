import pytest

from app.datakit.resolve import DataKit


@pytest.fixture(scope="module")
def kit():
    return DataKit()


def names(res):
    return [c.name for c in res.candidates]


def test_exact_abbreviation_resolves(kit):
    r = kit.resolve("test", "cbc")
    assert r.status == "resolved" and r.best.name == "Complete Blood Count"


def test_spelling_variants(kit):
    assert kit.resolve("test", "shugar fasting").best.name == "Blood Glucose Fasting"
    r = kit.resolve("test", "tsh")
    assert r.status == "resolved" and r.best.name == "TSH"
    assert kit.resolve("test", "thyroid").status == "ambiguous"       # TSH, thyroid profile, thyroid ultrasound...


def test_vitamin_d_is_ambiguous(kit):
    r = kit.resolve("test", "vitamin d")
    assert r.status == "ambiguous"
    assert set(names(r)) >= {"Vitamin D Total", "Vitamin D 25 Hydroxy"}


def test_sugar_is_ambiguous_between_three(kit):
    r = kit.resolve("test", "sugar")
    assert r.status == "ambiguous"
    assert len(r.candidates) == 3


def test_unknown_test(kit):
    assert kit.resolve("test", "xyzzy quark").status == "none"


def test_with_filler_words(kit):
    assert kit.resolve("test", "cbc ka test karwana hai").best.name == "Complete Blood Count"


def test_split_and_resolve_many(kit):
    res = kit.resolve_many("test", "cbc aur lipid profile")
    assert [r.best.name for r in res] == ["Complete Blood Count", "Lipid Profile"]


def test_spelled_out(kit):
    r = kit.resolve_spelled("test", "c b c")
    assert r.best.name == "Complete Blood Count"


def test_area_inside_address(kit):
    r = kit.resolve("area", "gulshan e iqbal block 13 d ghar number 45")
    assert r.best.name == "Gulshan-e-Iqbal"
    assert r.status in ("resolved", "confirm")


def test_area_gulshan_alone_is_ambiguous(kit):
    r = kit.resolve("area", "gulshan")
    assert r.status == "ambiguous"
    assert {"Gulshan-e-Iqbal", "Gulshan-e-Maymar"} <= set(names(r))


def test_uncovered_area_fact(kit):
    r = kit.resolve("area", "korangi")
    assert r.best.name == "Korangi"
    assert kit.facts("area", r.best.id, ["covered"]) == {"covered": "no"}


def test_facts(kit):
    r = kit.resolve("test", "cbc")
    f = kit.facts("test", r.best.id, ["price", "fasting"])
    assert f == {"price": "800", "fasting": "no"}


def test_describe(kit):
    r = kit.resolve("test", "vitamin d")
    c = r.candidates[0]
    assert kit.describe("test", c, r.candidates[1:])  # non-empty label


def test_recognizer_garbles_from_real_calls_resolve(kit):
    assert kit.resolve("test", "seebeesee").best.name == "Complete Blood Count"
    assert kit.resolve("test", "elaephti").best.name == "Liver Function Test"
    assert kit.resolve("test", "echbi a one si").best.name == "HbA1c"
    assert kit.resolve("test", "kowid peeseeaar").best.name == "COVID PCR"


def test_single_letter_names_are_not_dropped(kit):
    assert kit.resolve("test", "hepatitis a").best.name == "Hepatitis A IgM"
    assert kit.resolve("test", "hepatitis b").best.name == "Hepatitis B Surface Antigen"
    assert kit.resolve("test", "hepatitis").status == "ambiguous"


def test_generic_names_with_several_tests_are_ambiguous(kit):
    for q in ("dengue", "typhoid", "ct scan", "mri", "ultrasound", "sugar"):
        assert kit.resolve("test", q).status == "ambiguous", q


def test_imaging_is_not_home_collectable(kit):
    r = kit.resolve("test", "mri brain")
    assert kit.facts("test", r.best.id, ["home_collection"]) == {"home_collection": "no"}
    assert kit.facts("test", kit.resolve("test", "cbc").best.id, ["home_collection"]) == {"home_collection": "yes"}


def test_dha_is_ambiguous_between_cities(kit):
    r = kit.resolve("area", "dha")
    assert r.status == "ambiguous" and {"DHA Lahore", "DHA Karachi"} <= set(names(r))
    assert kit.resolve("area", "johar town block c ghar 25").best.name == "Johar Town Lahore"


def test_sample_list_sizes(kit):
    assert len(kit._entities["test"].records) >= 120 and len(kit._entities["area"].records) >= 60
