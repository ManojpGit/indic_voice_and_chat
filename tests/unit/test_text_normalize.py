from __future__ import annotations

from src.pipeline.text_normalize import DEFAULT_PRONUNCIATIONS, apply_pronunciations


def test_rewrites_known_terms_to_devanagari() -> None:
    out = apply_pronunciations("WhatsApp par link bhejun? Casino bhi hai.")
    # Mispronounced English terms are gone, replaced by Devanagari.
    assert "WhatsApp" not in out and "Casino" not in out and "link" not in out
    assert DEFAULT_PRONUNCIATIONS["WhatsApp"] in out
    assert DEFAULT_PRONUNCIATIONS["Casino"] in out
    assert DEFAULT_PRONUNCIATIONS["link"] in out
    # Surrounding text is preserved.
    assert "bhejun" in out and "bhi hai" in out


def test_case_insensitive_whole_word_only() -> None:
    out = apply_pronunciations("whatsapp WHATSAPP Whatsapp")
    assert out.count(DEFAULT_PRONUNCIATIONS["WhatsApp"]) == 3
    # 'app' must not match inside another word, only standalone.
    assert apply_pronunciations("happy apple") == "happy apple"
    assert DEFAULT_PRONUNCIATIONS["app"] in apply_pronunciations("open the app")


def test_empty_and_no_match_passthrough() -> None:
    assert apply_pronunciations("") == ""
    assert apply_pronunciations("sirf hindi text hai") == "sirf hindi text hai"


def test_extra_overrides_merge_over_defaults() -> None:
    out = apply_pronunciations("Khelo Aviator aur ZyxBrand", extra={"ZyxBrand": "ज़िक्सब्रांड"})
    assert "ज़िक्सब्रांड" in out
    assert DEFAULT_PRONUNCIATIONS["Aviator"] in out
