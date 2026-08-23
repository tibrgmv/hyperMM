from hypermm import config


def test_instruments_are_the_three_covered_by_the_dataset():
    assert config.INSTRUMENTS == ("BTC", "ETH", "SOL")


def test_ensure_data_dirs_is_idempotent(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", processed)

    config.ensure_data_dirs()
    assert raw.is_dir()
    assert processed.is_dir()

    config.ensure_data_dirs()
