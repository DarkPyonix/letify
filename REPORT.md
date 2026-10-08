# IPython 셀 의존성 전송 수정

IPython 사용자 세션 모듈을 값으로 직렬화하여 다른 셀의 클래스와 함수를 런타임으로 전송하도록 수정했다. `Env.ship()` 설정이나 런타임의 IPython 설치 없이 persistent와 one-shot 경로에서 동작한다.

## 재현

Python `3.13.15`, IPython `9.17.1`에서 브라우저 없이 `InteractiveShell.run_cell()`을 두 번 실행했다. 기본 `InteractiveShell()`에서는 클래스와 함수의 `__module__`이 모두 `__main__`이고 Local 호출 결과는 `15`였다. 기본 IPython이 모든 셀에 별도 모듈 이름을 부여한다는 가정은 이 환경에서 확인되지 않았다.

실패 조건은 IPython이 지원하는 `user_module=ModuleType("notebook_session")`으로 명시적으로 구성했다. 이는 이름이 다른 사용자 네임스페이스를 재현하며, 기본 Jupyter 또는 Colab에서 자동으로 발생한다고 확인한 것은 아니다.

```bash
uv run --with ipython python - <<'PY'
import tempfile
from pathlib import Path
from types import ModuleType
from IPython.core.interactiveshell import InteractiveShell
import letify

shell = InteractiveShell(user_module=ModuleType("notebook_session"))
shell.user_ns["let"] = letify.Launcher(
    Path(tempfile.mkdtemp()) / ".letify", home=False, announce=False
)
shell.run_cell(
    "class Offset:\n"
    "    def apply(self, x):\n"
    "        return x + 10",
    store_history=True,
).raise_error()
shell.run_cell(
    '@let.function(device=let.providers.local.CPU, host="remote")\n'
    "def score(x):\n"
    "    return Offset().apply(x)",
    store_history=True,
).raise_error()
print(shell.user_ns["Offset"].__module__)
print(shell.user_ns["score"](5))
PY
```

수정 전 클래스의 `__module__`은 `notebook_session`이었다. Local worker 호출은 `RemoteError: ModuleNotFoundError: No module named 'notebook_session'`로 실패했다.

## 원인과 수정

`letify/declare/function.py:57`는 선언 함수 자체를 `self.fn`에 보관한다. 모듈의 함수 이름은 데코레이터가 반환한 `Function` 객체를 가리키므로, cloudpickle은 원래 함수를 이름으로 찾지 못하고 값으로 전송한다. 반면 같은 세션의 클래스나 일반 함수는 모듈에서 이름으로 찾을 수 있어 참조로 전송된다. 런타임에는 해당 세션 모듈이 없어 역직렬화가 실패한다.

`letify/runtime/session.py:259`는 `env.ship_modules`만 명시적으로 등록했고, `letify/protocol/codec.py:54`부터 `:62`까지는 그 목록의 모듈을 import하여 cloudpickle에 등록했다. IPython 사용자 모듈은 자동 등록되지 않았다. `letify/protocol/worker.py:237`의 `cloudpickle.loads()`에서 오류가 발생했다.

`letify/protocol/worker.py:216`의 `__main__` 검사는 호출 내부에서 생성하는 자식 프로세스의 직렬화 처리다. 이번 실패는 호출 본문 실행 이전에 발생하므로 이 코드는 변경하지 않았다.

`letify/protocol/codec.py:65`에 `_ship_notebook_module()`을 추가했다. 함수의 globals에 있는 `get_ipython` 바인딩에서 세션을 찾고, `user_module.__dict__`이 함수의 globals와 같은 객체인지 확인한 뒤 사용자 모듈만 값으로 전송하도록 등록한다. `dumps_call()`의 `:76`과 `dumps_call_parts()`의 `:88`에서 호출한다. IPython을 import하거나 세션 객체를 payload에 추가하지 않는다. 외부 패키지의 등록 목록은 확장하지 않는다.

## 검증

`tests/test_notebook.py`는 첫 셀에서 `Offset`과 이를 사용하는 `adjusted`를 정의하고, 둘째 셀에서 `score`를 선언한다. `__main__`과 `letify_notebook_session`을 persistent와 one-shot 경로에서 각각 실행하는 `4`개 조건을 검사한다. persistent 조건은 실제 Local provider worker를 사용하고, one-shot 조건은 기존 `local_one_shot_runner()`로 실제 드라이버 프로세스를 실행한다. 결과 `15`와 호출자와 다른 process ID를 검증한다. IPython이 없으면 기존 선택적 의존성 방식인 `pytest.importorskip()`으로 건너뛴다.

수정 전 `uv run --with ipython pytest -q -p no:cacheprovider tests/test_notebook.py` 결과는 `2 failed, 2 passed in 0.68s`였다. 실패한 두 조건의 traceback은 모두 `ModuleNotFoundError`와 `letify_notebook_session`을 포함했다.

수정 후 `uv run --with ipython pytest -q -p no:cacheprovider tests/test_notebook.py tests/test_protocol.py` 결과는 `47 passed in 2.47s`였다. Ruff lint와 format 검사도 통과했다. 기존 assertion을 삭제하거나 완화하지 않았고 sleep을 추가하지 않았다.

전체 테스트는 로컬 가상 환경에 IPython과 CPU PyTorch를 설치한 뒤 요청한 명령으로 한 번 실행했다. `pyproject.toml`과 `uv.lock`은 변경하지 않았다.

```bash
uv run pytest -q -p no:cacheprovider
```

결과는 `1469 passed, 3 skipped, 7 warnings in 343.49s (0:05:43)`이며 `0 failed`다. develop 기준인 `1465 passed, 0 failed, 3 skipped`보다 새 회귀 조건 `4`개가 추가로 통과했고 skipped 수는 같다. 경고는 다중 스레드 프로세스의 `fork()`와 PyTorch tensor의 scalar 변환에 관한 것이다.

## 커밋

- `fdfc314`: `Docs: Specify shipping definitions across IPython cells`
- `344b98b`: `Test: Reproduce missing dependencies from IPython cells`
- `31f3ed3`: `Fix: Ship the IPython session module with each call`

push, develop 병합, 브랜치 전환은 수행하지 않았다.
