from app.faq import FaqBook


def test_faq_answers_and_fills_placeholders():
    fb = FaqBook()
    m = fb.match("lab kitne baje khulti hai")
    assert m.status == "answer" and "khuli" in m.text
    m = fb.match("home sampling ka charge kitna hai")
    assert m.entry["id"] == "faq_home_fee" and "300" in m.text


def test_faq_no_match():
    fb = FaqBook()
    assert fb.match("mujhe cbc karwana hai").status != "answer"
    assert fb.match("xyzzy").status == "none"


def test_faq_checks_clean():
    assert FaqBook().check() == []


def test_new_sample_faqs_match_real_call_topics():
    fb = FaqBook()
    for q, fid in [("online report kaise milegi", "faq_report_online"), ("report whatsapp par mil sakti hai", "faq_report_whatsapp"),
                   ("mri par discount milta hai", "faq_imaging_discount"), ("doctor sahab se baat karni hai", "faq_doctor_consult"),
                   ("sialkot mein home sampling hoti hai", "faq_home_cities"), ("slip number yaad nahi report chahiye", "faq_report_lost_slip")]:
        m = fb.match(q)
        assert m.status == "answer" and m.entry["id"] == fid, (q, m.status, m.entry and m.entry["id"], round(m.score, 2))


def test_handoff_entries_flagged_and_samples_marked():
    fb = FaqBook()
    assert fb.entry_by_id("faq_report_delay").get("handoff")
    assert fb.sample_count() == len(fb.entries)
