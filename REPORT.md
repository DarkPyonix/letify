# Colab TCP punch 회귀 수정 보고

`tcp_punch` rendezvous에서 wake 잠금 대기를 제거했다. wake끼리의 직렬화는 유지하며, 잠금이 사용 중이면 이번 wake를 즉시 건너뛴다. tailcat 설치는 해당 전략의 endpoint 요청 시점으로 옮겼다. `target()`은 원격 명령 없이 반환하므로 설치가 race 시작을 막지 않는다. `10.0 MiB/s` throughput floor는 그대로 유지했다. 실제 계정에 요청하지 않았으며 실계정 연결 복구 여부는 아직 검증하지 않았다.

## 확인한 원인

아래 원인 위치는 수정 전 `v1.2.1` 기준이다.

1. `letify/providers/colab.py:283`에서 `_exec` 전체를 `account_lock`으로 감싼다. `letify/providers/colab.py:333`의 `ColabRendezvous` callback도 `_exec`를 호출한다. `letify/providers/colab_keepalive.py:63`의 wake도 같은 계정 디렉터리의 잠금을 획득하고 명령 완료까지 보유한다. `letify/providers/colab_keepalive.py:24`의 성공 주기는 `60.0 s`, `letify/providers/colab_keepalive.py:26`의 명령 timeout은 `120.0 s`이다. 잠금 대기는 subprocess timeout 바깥에 있어 rendezvous의 timeout으로도 제한되지 않는다. `letify/transport/strategies.py:177`은 원격 실행 전에 `start_at`을 정한다. 따라서 wake 뒤에서 기다리면 합의된 punch 시작 시각을 놓칠 수 있다. fake wake가 잠금을 보유한 동안 rendezvous가 완료되지 않는 현상을 테스트로 재현했다.
2. `letify/providers/colab.py:357`은 `target()` 안에서 tailcat 설치를 `180 s` timeout으로 실행한다. 같은 `_exec` 잠금 경로를 사용하며 race보다 먼저 실행된다. 설치 실패도 모든 연결 시도를 중단시킨다. 이것은 race 시작 지연이고, 그 자체로 이미 정해진 `start_at`을 소모하는 것은 아니다. 두 경로의 공통 잠금과 설치가 연결 준비를 직렬화한다는 점을 확인했다.

`git log v1.1.1..v1.2.0 -- letify/transport/ letify/providers/colab.py`에서 `1e1f849`의 도입과 후속 `2f553d6`을 확인했다. `git diff v1.1.1..HEAD -- letify/transport/ letify/providers/colab.py`도 확인했다. `v1.1.1`의 `_exec`에는 이 잠금이 없고 `target()`에는 원격 tailcat 설치가 없었다.

사용자가 제공한 `2026-10-08 05:01~05:02 UTC`, `letify-a100-b34279` 측정에서 tailcat은 `8.2 s`에 연결되고 `93.3 ms`, up `4.0 MiB/s`, down `0.1 MiB/s`였다. 이 링크의 floor 거절은 올바르다. `tcp_punch`는 `20.7 s` 후 취소되었다. 코드와 fake 테스트는 잠금으로 punch가 지연될 수 있음을 확인하지만, 해당 실계정 실행의 잠금 보유 시간까지 증명하지는 않는다.

## 변경과 결정

- `letify/providers/colab.py:280`: `_exec`의 wake 잠금을 제거했다. subprocess timeout과 CLI reply timeout의 기존 오류 진단은 유지했다. wake끼리의 CLI session state 충돌을 막는 잠금에 별도 작업인 rendezvous를 넣을 필요가 없다는 결정이다. 이미 실행 중인 wake가 `120.0 s` 동안 잠금을 보유해도 connection은 그 잠금을 요청하지 않는다.
- `letify/providers/colab_keepalive.py:31`: `account_lock`에 nonblocking 획득을 추가했다. `letify/providers/colab_keepalive.py:73`의 wake는 대기 시간이 `0 s`인 획득을 사용한다. 실패하면 명령을 실행하지 않고 `wake_skipped`, `account_busy`를 history와 log에 기록한다. 다음 시도는 `60.0 s` 뒤다. 잠금을 얻은 wake는 명령 종료까지 보유하므로 두 wake가 동시에 실행되지 않는다. 기존 명령 실패의 `5.0 s` 재시도와 `120.0 s` timeout도 유지했다.
- `letify/providers/colab.py:330`과 `letify/transport/rendezvous.py:98`: tailcat 요청에만 설치 callback을 실행한다. `letify/transport/strategies.py:214`에서 endpoint를 요청하는 race thread가 설치를 수행한다. `letify/providers/colab.py:366`의 `target()`에는 설치가 없다. 설치가 지연되거나 실패해도 punch는 기다리지 않는다. SHA-256 검증과 오류 전달은 유지하고, 설치 실패 범위는 tailcat 전략으로 한정했다. 준비된 binary는 tailcat 요청에만 포함한다.

선택은 lazy 설치다. 위 측정의 tailcat은 해당 floor를 통과할 수 없지만 한 세션의 측정으로 모든 Colab 링크가 항상 같은 성능이라고 단정하지 않았다. provider 전체에서 tailcat을 없애는 대신, 사용할 전략이 실제 요청할 때만 설치하고 독립적인 punch를 막지 않도록 했다. floor를 낮추거나 실패한 probe를 허용하는 변경은 없다. 결정과 이유는 `docs/SPEC.md:1376`의 Runtime tools, Rendezvous scheduling, Keep-alive supervision에 기록했다.

## 추가 확인과 배제

- `letify/transport/pipeline.py:241`: lone strategy probe는 `strategy.assume()`으로 연결한 뒤 실행된다. rendezvous 전의 probe 대기는 추가하지 않는다.
- `letify/transport/pipeline.py:272`와 `letify/transport/pipeline.py:296`: network fingerprint, cache 읽기, cached strategy 연결과 probe는 `v1.1.1`에도 race 전에 있었다. cached tailcat을 먼저 검증하면 race 시작이 늦어질 수 있지만 새로 추가된 순서가 아니다. punch의 `start_at`은 그 뒤 해당 시도가 시작될 때 정한다.
- `letify/transport/pipeline.py:32`와 `letify/transport/pipeline.py:96`: `3600.0 s` TTL과 cache version `2`는 오래되거나 다른 형식인 cache를 거절한다. 새로운 원격 대기 단계는 없다. floor는 연결 후 probe 결과에 적용된다.
- `letify/transport/pipeline.py:360`의 race는 전략별 thread를 먼저 시작하고 결과를 모은다. `letify/transport/pipeline.py:443`의 probe는 race 결과가 정해진 뒤 실행된다. tailcat probe가 진행 중인 punch rendezvous 앞에 끼어드는 순서는 아니다. `2.0 s` grace와 cancellation 순서는 이번에 바꾸지 않았다. 빠르게 연결된 tailcat이 punch보다 먼저 grace를 시작할 가능성은 기존 동작이며 실계정 검증에서 확인해야 한다.
- `letify/transport/nat.py:203`: `connect_ex`의 즉시 실패를 판별해 socket을 닫고 `0.2 s` 뒤 다시 시도하는 변경이다. 이는 잘못 연결된 socket으로 hello를 쓰는 문제를 방지한다. rendezvous 전에 새 대기를 추가하지 않으며 probe나 cache를 읽지 않는다. 실제 NAT에서의 영향은 fake 검사만으로 배제할 수 없다.
- `letify/transport/strategies.py:171`의 punch 순서, `10.0 s` lead와 `30.0 s` rendezvous command timeout은 `v1.1.1`과 같다. strategies의 다른 diff는 ReverseSSH 임시 디렉터리 위치이며 Colab punch에 적용되지 않는다.

## 테스트 증거

spec 커밋 후 테스트를 먼저 작성했다. 다음 선택 실행에서 구현 수정 전에 `4 failed, 226 deselected`를 확인했고, 구현 수정 후 같은 선택 실행은 `4 passed, 226 deselected`였다.

```text
uv run --offline pytest -q tests/test_providers.py tests/test_colab_keepalive.py -k 'verified_tailcat or holds_the_account_lock or during_tailcat_installation or skips_a_busy_account'
```

- `tests/test_providers.py:2639`: 기존 설치 검증 테스트를 lazy endpoint 경로로 옮겼다. `target()`의 원격 호출 없음, exec command, SHA-256 source, 검증된 경로, 설치 오류의 `RuntimeFailure`를 확인한다. 기존 검증을 삭제하거나 약화하지 않았다.
- `tests/test_providers.py:2803`: fake wake 명령을 Event로 멈춰 실제 account 잠금을 보유시킨다. 그동안 rendezvous가 끝나야 한다. 수정 전에는 answers가 비어 실패했다. 최종 테스트의 요청 kind는 `tcp_punch`다.
- `tests/test_providers.py:2849`: `target()`의 호출 없음과 설치 중 별도 punch rendezvous 완료를 확인한다. 수정 전에는 target 구성 중 설치 호출로 실패했다. 수정 후에는 설치를 Event로 멈춰도 punch가 응답한다.
- `tests/test_colab_keepalive.py:104`: 잠금이 사용 중인 동안 wake가 `60.0 s`를 반환하고 subprocess 호출 없이 skip을 기록해야 한다. 수정 전에는 잠금 뒤에서 기다려 실패했다.

기존 wake 잠금 보유 테스트도 그대로 통과했다. fake는 기존 `patch_run`을 사용했고 sleep은 추가하지 않았다. 별도 관련 실행은 `230 passed`, transport 실행은 `53 passed`였다. 테스트 최초 수집 시 import 편집 오류를 바로잡았고, 존재하지 않는 `tests/test_rendezvous.py`를 지정한 실행은 테스트를 수행하지 않아 올바른 경로로 다시 실행했다. 이 둘은 동작의 red 증거로 계산하지 않았다.

전체 suite는 `torch`가 없는 worktree `.venv`에서 먼저 시작했으나 forwarding 검사가 빠지는 것을 확인하고 중단했다. 중단 시점 결과는 `1123 passed, 20 skipped`, `KeyboardInterrupt`, `191.29 s`이며 완료된 전체 결과로 계산하지 않았다. `torch`가 있는 기존 환경으로 전환하고 `PYTHONPATH`를 worktree로 지정했다. 실제 import 경로도 이 worktree의 `letify/__init__.py`임을 확인했다. 전체 suite를 완료하는 명령은 다음과 같다.

```text
PYTHONPATH="$PWD" /workspace/brew/letify/.venv/bin/python -m pytest -q
```

완료된 전체 suite 1회 결과는 `1578 passed, 7 skipped, 4 failed, 7 warnings in 397.18 s`이다. 사용자가 제공한 main checkout 기준은 `1579 passed, 7 skipped, 0 failed`이다. 새 테스트 3개를 포함한 worktree의 전체 대상은 `1582`개와 skipped `7`개다.

실패 4개는 `tests/test_runtime.py:674`, `tests/test_runtime.py:1981`, `tests/test_runtime.py:2164`의 두 parameter 사례다. 모두 설치된 project `.venv`에서 import해야 한다는 기존 assertion에서 실패했다. 내가 전체 실행에 추가한 `PYTHONPATH`가 자식 worker에도 전달되어 `letify/__init__.py`를 worktree에서 import하게 한 테스트 환경 오류다. Colab 변경과 관계없는 Local 환경 검증이며 코드나 assertion을 바꾸지 않았다.

`PYTHONPATH`를 제거한 아래 재검증은 `4 passed in 6.78 s`였다. 이 명령에서도 parent의 실제 `letify.__file__`은 worktree 경로임을 별도로 확인했다. `python -m pytest`의 현재 디렉터리 import 순서만으로 충분하며 전역 `PYTHONPATH` 지정은 필요하지 않았다.

```text
env -u PYTHONPATH /workspace/brew/letify/.venv/bin/python -m pytest -q tests/test_runtime.py::test_a_default_env_is_synced_with_uv_and_the_worker_runs_from_the_project_venv tests/test_runtime.py::test_a_provider_with_an_env_root_builds_the_venv_there_but_the_uv_cache_in_the_workspace tests/test_runtime.py::test_an_ephemeral_account_restores_an_environment_without_declared_volumes
```

전체 실행에서 통과한 `1578`개와 환경을 바로잡아 재검증한 `4`개를 합하면 전체 대상 `1582`개가 통과했다. 이것을 단일 전체 실행의 `1582 passed, 7 skipped, 0 failed` 결과라고 쓰지는 않는다. 완료된 전체 실행은 1회이며 실패했던 4개만 다시 실행했다. 환경 수정 후 재검증에서 남은 실패는 `0`개다.

변경한 Python 파일의 `ruff check`는 통과했다. 전체 `ruff check .`에는 변경 전부터 있는 `letify-core/build.py:3`의 `E501`이 남아 있다. 추가한 테스트의 줄 길이 오류는 별도 `Test:` 커밋으로 수정했다.

## 실계정에서 남은 확인

실제 계정에는 어떤 네트워크 요청도 보내지 않았다. `uv` 실행도 `--offline`을 사용했다. daemon subprocess 테스트는 로컬 Python의 `pass`만 실행한다.

다음 실계정 실행에서는 동일한 account와 network에서 `tcp_punch`가 합의된 시각에 시작하고 연결되는지, wake 실행과 rendezvous가 CLI 내부 session state 충돌 없이 겹칠 수 있는지 확인해야 한다. lazy tailcat 설치가 notebook kernel 내부에서 punch 실행을 지연시키는지도 확인해야 한다. Python 잠금 대기는 제거했지만 외부 CLI와 kernel 내부 동시성은 fake로 증명할 수 없다. 실제 NAT 성공률과 `nat.py`의 dial 재시도도 함께 확인해야 한다.

연결 이후 양방향 throughput과 RTT가 기존 floor를 통과하는지도 필요하다. `v1.1.1`의 `5 to 24 MiB/s` 중 `5 MiB/s` 링크는 현재 기본 `10.0 MiB/s` floor를 통과하지 않는다. punch 연결 복구와 floor 통과는 구분해서 검증해야 한다. tailcat의 down `0.1 MiB/s` 거절은 계속 유지된다.

현재 branch는 `fix/punch-is-not-starved`다. branch 변경, push, develop merge는 하지 않았다. spec, 테스트, 구현은 각각 `85000ee`, `f879958`, `9fde55e`에 커밋했다. 테스트 줄 길이 수정은 `32a5048`이다. attribution trailer는 추가하지 않았다.
