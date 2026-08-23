import pytest

from hypermm.ingest.codes import Status, Tif, check_codes

STATUSES_CSV = """badAloPxRejected,0
open,1
canceled,2
perpMarginRejected,3
iocCancelRejected,4
filled,5
minTradeNtlRejected,6
reduceOnlyCanceled,7
reduceOnlyRejected,8
triggered,9
scheduledCancel,10
siblingFilledCanceled,11
selfTradeCanceled,12
marginCanceled,13
vaultWithdrawalCanceled,14
perpMaxPositionRejected,15
liquidatedCanceled,16
oracleRejected,17
"""

TIFS_CSV = """Alo,0
Gtc,1
Ioc,2
null,3
FrontendMarket,4
LiquidationMarket,5
"""


@pytest.fixture
def mapdir(tmp_path):
    (tmp_path / "statuses.csv").write_text(STATUSES_CSV)
    (tmp_path / "tifs.csv").write_text(TIFS_CSV)
    return tmp_path


def test_codes_used_downstream_have_the_documented_values():
    assert Status.BAD_ALO_PX_REJECTED == 0
    assert Status.OPEN == 1
    assert Status.CANCELED == 2
    assert Status.FILLED == 5
    assert Tif.ALO == 0


def test_accepts_the_published_lookup_tables(mapdir):
    check_codes(mapdir)


def test_rejects_a_renumbered_status(mapdir):
    (mapdir / "statuses.csv").write_text(STATUSES_CSV.replace("filled,5", "filled,6"))
    with pytest.raises(ValueError, match="statuses.csv"):
        check_codes(mapdir)


def test_rejects_a_renamed_status(mapdir):
    (mapdir / "statuses.csv").write_text(STATUSES_CSV.replace("filled,5", "matched,5"))
    with pytest.raises(ValueError, match="FILLED"):
        check_codes(mapdir)


def test_rejects_an_unknown_extra_code(mapdir):
    (mapdir / "statuses.csv").write_text(STATUSES_CSV + "somethingNew,18\n")
    with pytest.raises(ValueError, match="18"):
        check_codes(mapdir)


def test_tolerates_a_trailing_blank_line(mapdir):
    (mapdir / "statuses.csv").write_text(STATUSES_CSV + "\n")
    check_codes(mapdir)


def test_the_literal_null_tif_label_survives_reading(mapdir):
    check_codes(mapdir)
    assert Tif(3).name == "NULL"
