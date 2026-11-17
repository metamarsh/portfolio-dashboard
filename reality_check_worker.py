"""
reality_check_worker.py  (Backtest Engine; added November 16, 2026)

Worker processes for the Reality Check section of Adaptive Portfolio Dashboard.py. The section repeats the
backtest on dozens of nearby settings, and one run_backtest call takes a few seconds, so on a PC the dashboard
hands those runs to a pool of worker processes (rc_run_parallel in the dashboard). Each worker loads the
dashboard's own engine code once and then runs the backtests it is given, so every figure still comes from
run_backtest itself.

A worker loads the dashboard file's non-UI half only (everything above the sidebar), with a stand-in for
Streamlit, so no page is drawn, nothing is downloaded and no file is written. The prices arrive as a
temporary pickle the dashboard writes for the pool and deletes afterward.

If the pool cannot start, the dashboard runs the same backtests one at a time instead.
"""
import io
import pickle
import sys
import types

UI_MARKER = 'st.sidebar.title("Backtest Parameters")'
_ENGINE = None
_DATA = None


class _Stub:
    def __call__(self, *a, **k):
        return _Stub()

    def __getattr__(self, name):
        return _Stub()

    def __enter__(self):
        return _Stub()

    def __exit__(self, *a):
        return False


def load_engine(path):
    """Exec the dashboard's non-UI half into a namespace, with a Streamlit stand-in."""
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    cut = next(i for i, ln in enumerate(lines) if UI_MARKER in ln)
    head = "\n".join(lines[:cut])
    stub = types.ModuleType("streamlit")
    stub.__getattr__ = lambda name: _Stub()
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub
    ns = {"__name__": "reality_check_engine", "__file__": path}
    try:
        exec(compile(head, path, "exec"), ns)
    finally:
        if saved is not None:
            sys.modules["streamlit"] = saved
        else:
            sys.modules.pop("streamlit", None)
    return ns


def init(engine_path, data_path):
    """Pool initializer: load the engine and the run's prices once per worker."""
    global _ENGINE, _DATA
    _ENGINE = load_engine(engine_path)
    with open(data_path, "rb") as fh:
        _DATA = pickle.load(fh)


def run(job):
    """One neighbor backtest: (key, params) -> (key, what the panel keeps of the run)."""
    key, params = job
    try:
        res, _trades, deq = _ENGINE["run_backtest"](_DATA["daily"], _DATA["monthly"], params)
        entry = _ENGINE["rc_entry"](res, deq, _DATA["rf"])
    except Exception:
        entry = _ENGINE["rc_entry"](None, None, _DATA["rf"])
    return key, entry
