from src.pipeline.text_normalize import normalize_currency


def test_rupee_symbol():
    assert normalize_currency("आप सिर्फ ₹100 से शुरू करें") == "आप सिर्फ 100 रुपये से शुरू करें"


def test_rupee_symbol_with_space():
    assert normalize_currency("₹ 500 का बोनस") == "500 रुपये का बोनस"


def test_rs_word_and_dot():
    assert normalize_currency("Rs 100 se shuru") == "100 रुपये se shuru"
    assert normalize_currency("Rs. 250") == "250 रुपये"


def test_commas_stripped():
    assert normalize_currency("₹1,000 deposit") == "1000 रुपये deposit"


def test_spelled_out_untouched():
    # Already-spoken forms must not be altered.
    assert normalize_currency("100 रुपये से शुरू") == "100 रुपये से शुरू"
    assert normalize_currency("कोई amount नहीं") == "कोई amount नहीं"


def test_empty():
    assert normalize_currency("") == ""
