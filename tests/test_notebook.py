"""IPython cell dependencies through the real Local worker, per spec "Call protocol"."""

from __future__ import annotations

import os
import sys
from types import ModuleType

import cloudpickle
import pytest
from conftest import local_one_shot_runner

from letify.runtime.channel import OneShotChannel


@pytest.mark.parametrize("module_name", ["__main__", "letify_notebook_session"])
@pytest.mark.parametrize("channel_kind", ["persistent", "one-shot"])
def test_a_call_ships_definitions_from_another_ipython_cell(
    let, cpu, monkeypatch, module_name, channel_kind
) -> None:
    ipython = pytest.importorskip("IPython.core.interactiveshell")
    module = ModuleType(module_name)
    # Record the original module before IPython installs its session namespace.
    monkeypatch.setitem(sys.modules, module_name, module)
    shell = ipython.InteractiveShell(user_module=module)
    shell.user_ns.update(let=let, cpu=cpu)
    try:
        shell.run_cell(
            "import os\n"
            "class Offset:\n"
            "    def apply(self, value):\n"
            "        return value + 10\n"
            "def adjusted(value):\n"
            "    return Offset().apply(value)\n",
            store_history=True,
        ).raise_error()
        shell.run_cell(
            '@let.function(device=cpu, host="remote")\n'
            "def score(value):\n"
            "    return adjusted(value), os.getpid()\n",
            store_history=True,
        ).raise_error()
        assert shell.user_ns["Offset"].__module__ == module_name
        score = shell.user_ns["score"]
        if channel_kind == "persistent":
            value, worker_pid = score(5)
        else:
            channel = OneShotChannel(local_one_shot_runner(), name="notebook")
            (value, worker_pid), _logs = channel.call(score.fn, (5,), {})
        assert value == 15
        assert worker_pid != os.getpid()
    finally:
        shell.restore_sys_module_state()
        if module_name in cloudpickle.list_registry_pickle_by_value():
            cloudpickle.unregister_pickle_by_value(module)
