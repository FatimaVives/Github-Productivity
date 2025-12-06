from data import load_raw
def test_load_has_expected_cols():
    df = load_raw()
    assert "commits_count" in df.columns
