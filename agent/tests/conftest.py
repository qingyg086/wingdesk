"""
pytest 公共夹具。

测试使用独立的 MySQL 库 airline_test（每次需要时从 data/airline/*.json 重新导入），不会动正式库 airline。
运行：在项目根目录执行  pytest agent/tests -q
"""

import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR / "database"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_airline import import_all                        # noqa: E402
from backends.mysql_backend import (MySQLBackend, taubench_certificate_id,  # noqa: E402
                                    taubench_reservation_id)
from taubench_ref import fresh_data, load_reference_tools      # noqa: E402

TEST_DB = "airline_test"


@pytest.fixture(scope="session")
def ref_tools():
    return load_reference_tools()


@pytest.fixture(scope="session")
def _imported_once():
    assert import_all(TEST_DB, verbose=False), "导入测试库失败"


@pytest.fixture(scope="session")
def backend(_imported_once):
    """用 τ-bench 风格的编号生成器，这样新订单号、代金券号也能和原版逐字比对。"""
    return MySQLBackend(TEST_DB, pool_size=4, new_reservation_id=taubench_reservation_id,
                        new_certificate_id=taubench_certificate_id)


@pytest.fixture(scope="module")
def ro(backend, ref_tools):
    """只读测试：每个测试文件开始时重新导入一次（之前的写测试可能改过数据），文件内共享。"""
    assert import_all(TEST_DB, verbose=False)
    return backend, fresh_data()


@pytest.fixture()
def rw(backend, ref_tools):
    """写测试：每个测试前重新导入测试库、拿一份新的原版数据，两边都从同一个初始状态开始。"""
    assert import_all(TEST_DB, verbose=False)
    return backend, fresh_data()
