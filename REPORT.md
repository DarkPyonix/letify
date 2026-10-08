# 연결 floor 회귀 수정 보고

측정된 링크가 모두 floor 미달이면 `ProviderUnavailable`로 연결을 거부하도록 수정했다. 측정하지 않은 fallback으로 우회하지 않는다. 공통 기본값은 양방향 `10 MiB/s`, RTT 상한 `300 ms`다. Spec, 테스트, 코드를 별도 커밋으로 기록했다. 실제 계정에는 요청하지 않았고 push, merge, branch 전환도 하지 않았다.

## 실패가 발생한 경로

보고 대상은 `letify-a100-330e87`, `2026-10-08 04:27`부터 `04:30 UTC`다. `keep_alive_started`와 원격 `tailcat` 설치 성공은 정상 동작으로 유지했다.

`letify/transport/pipeline.py`의 `Pipeline._choose`는 연결된 링크를 `probed`와 `unprobed`로 나눴다. `tailcat`의 측정값은 RTT `177.4 ms`, upload `2.6 MiB/s`, download `0.1 MiB/s`였다. 기존 throughput floor `5 MiB/s`를 통과하지 못해 `probed`에서 제거됐다. 그런데 이후 분기는 `elif unprobed`를 `elif floored_out`보다 먼저 검사했다. 따라서 이미 연결된 `fallback`이 선택됐고 floor 실패 예외에는 도달하지 않았다. 아직 연결되지 않은 `tcp_punch`는 race 결정 시점에 취소됐다.

`docs/NETWORK.md`의 `colab exec` RTT는 `1.7`부터 `3.7 s`다. 이는 `177.4 ms`보다 약 `9.6`부터 `20.9`배 길다. 이 command 경로와 별도 bulk file API를 구분해야 한다. File API의 throughput을 `colab exec`의 throughput으로 해석할 수 없다. 측정에 실패한 링크보다 나은지 확인하지 않은 fallback이 floor 거부의 결과로 선택된 것이 회귀의 핵심이다.

## 선택한 동작과 floor

`docs/INTENT.md`의 원칙에 맞춰 경고 후 사용이 아니라 연결 거부를 선택했다. Floor 거부로 인해 더 느릴 수 있는 경로에 GPU 시간을 지출하는 일을 막는다. 모든 측정 링크가 탈락하면 남은 unprobed 링크도 닫고 측정값과 실패 기준을 포함한 `ProviderUnavailable`을 발생시킨다.

캐시 재검증에서도 같은 원칙을 적용했다. 캐시된 링크가 floor 미달이면 그 이유를 full race로 전달한다. Race에서 fallback만 남거나, 하나 남은 링크의 probe가 실패해도 앞선 floor 거부를 우회하지 못한다. 새로운 측정 링크가 기준을 통과하면 기존 선택 규칙을 적용한다. Floor 미달 측정이 전혀 없는 lone fallback과 명시적 `channel = "exec"`의 기존 동작은 유지된다. 이 예외도 `docs/SPEC.md`에 기록했다.

| 항목 | 값과 근거 |
| --- | --- |
| 공통 throughput 하한 | 양방향 각각 `10 MiB/s`. 사용자가 결정한 기준이다. Colab을 통과시키기 위해 낮추지 않았다. |
| 공통 RTT 상한 | `300 ms`. 변경할 측정 근거가 없으므로 유지했다. 보고된 `177.4 ms`는 이 조건을 통과한다. |
| 계정 override | `min_mib_per_s`, `max_rtt_ms`. `Shell.link_floor`가 공통 기본값에 계정 값을 적용한다. |
| Colab 최고 측정의 결과 | Upload `8.8 MiB/s`는 `10 MiB/s`보다 `1.2 MiB/s` 낮다. Download `10.7 MiB/s`만 통과하므로 전체 링크는 거부된다. |
| 보고된 A100 측정의 결과 | Upload `2.6 MiB/s`, download `0.1 MiB/s`가 모두 기준 미달이다. |

프로바이더별 floor 기본값이나 이를 선택하는 별도 로직은 없었다. `Shell`, `Colab`, `Tunnel`, `Elice`가 같은 기본값과 override를 사용하는지 테스트했다. `Modal`과 `Local`은 원래 이 연결 파이프라인에 포함되지 않는다.

Colab은 측정된 좋은 날의 upload도 기본 floor를 통과하지 못하므로 `min_mib_per_s`를 계정에 지정해야 사용할 수 있다. 이 결과를 `docs/SPEC.md`에 명시했다. `README.md`, `docs/locales/README_ko.md`의 계정 예시에는 느린 링크를 명시적으로 허용하는 `min_mib_per_s = 0.05`와 그 의미를 추가했다. 이 값은 프로바이더 기본값이 아니다. `docs/NETWORK.md`와 영어, 한국어 문제 해결 문서의 현재 floor 설명도 `10 MiB/s`로 맞췄다.

거부 메시지는 실패한 방향의 throughput 또는 RTT와 기준값을 유지하면서 두 설정 이름을 함께 표시한다. 예를 들어 보고된 링크는 `up 2.6 MiB/s`, `down 0.1 MiB/s`, 기준 `10.0 MiB/s`, `min_mib_per_s`, `max_rtt_ms`를 표시한다.

## Colab fallback timeout 조사

`Colab.fallback`은 `OneShotLink`를 만들고 `Shell.open_channel`은 이를 `OneShotChannel`로 연다. `OneShotChannel.start`는 상주 worker를 시작하지 않는다. 각 요청이 `Colab._exec`, `Colab._cli`, `colab exec`를 거쳐 notebook kernel 안에서 실행된다. 따라서 이 경로의 boot 실패는 persistent worker의 `HELLO` 대기 실패와 다르다.

`Runtime.prepare_workspace`, 환경 probe와 interpreter 검사 등 짧은 bootstrap 요청은 이미 `120 s`를 허용한다. 환경 설치는 `3600 s`다. `120 s`는 측정 RTT `1.7`부터 `3.7 s`의 약 `32.4`부터 `70.6`배다. 제한을 늘려야 한다는 근거는 찾지 못했다. Keep-alive와 kernel 명령은 기존 account lock으로 직렬화된다. Lock 대기는 subprocess timeout에 포함되지 않지만, 그것만으로 보고된 CLI reply timeout의 원인이라고 판단할 수 없다.

`Timeout waiting for reply` 문자열은 저장소의 letify 코드에 없다. CLI가 이 문자열을 stderr에 출력하고 실패하는 경우를 `patch_run`으로 재현했다. 실제 kernel이 응답하지 않은 이유나 CLI 내부 reply timeout 값은 이 worktree의 fake와 제공된 로그만으로 확정할 수 없다. 재현되지 않은 hang을 고쳤다고 주장하지 않고 오류 진단을 보강했다.

이제 CLI reply timeout은 session 이름과 `colab exec`를 명시하고 `the notebook kernel may be busy or not answering`을 가능한 원인으로 표시한다. 원래 command, stderr와 예외 원인을 보존한다. `subprocess.TimeoutExpired`도 요청 제한과 kernel 응답 가능 원인을 가진 `RuntimeFailure`로 변환한다. Runtime reclamation은 별도의 기존 증거 검사로 판단한다.

## Red와 green 증거

Spec 변경을 먼저 커밋하고 테스트를 작성한 뒤 실행했다. 기존 assertion을 삭제하거나 약화하지 않았다. 기존 default 값 assertion은 승인된 동작 변경에 맞춰 `5.0 * MIB`에서 `10.0 * MIB`로 강화했다. 테스트 통과를 위한 sleep을 추가하지 않았다.

| 테스트 | Red 증거 | Green 증거 |
| --- | --- | --- |
| `test_the_default_floor_is_ten_mib_per_second_and_three_hundred_ms` | 기존 `5 MiB/s` 값이 `10 MiB/s` assertion 실패 | 동일 assertion 통과, RTT `300 ms` 유지 |
| `test_floor_rejection_cannot_select_an_unprobed_fallback` | `2.6/0.1/177.4`와 `8.8/10.7/175.0` 두 경우 모두 예외 없이 fallback 선택 | 두 경우 모두 거부, 설정 이름과 측정값 표시, 두 링크 모두 닫힘 |
| `test_a_cached_floor_rejection_cannot_escape_to_an_unprobed_fallback` | 재측정 미달과 재연결 실패 두 경우 모두 fallback 선택 | 두 경우 모두 거부, 링크 종료, 기존 cache 보존 |
| `test_floor_rejections_name_both_account_overrides` | 기존 violations에 설정 이름 없음 | Upload, download, RTT 측정과 두 설정 이름 유지 |
| `test_shell_accounts_share_the_floor_and_keep_account_overrides` | `Shell`, `Colab`, `Tunnel`, `Elice` 네 경우 모두 기존 기본값 `5 MiB/s`로 실패 | 공통 `10 MiB/s`, `300 ms`와 override `0.05 MiB/s`, `500 ms` 모두 통과 |
| `test_colab_exec_reply_timeout_names_the_kernel_and_keeps_cli_evidence` | CLI 오류에 `notebook kernel` 진단 없음 | Kernel 가능한 원인과 원래 command, stderr 보존 |
| `test_colab_exec_command_timeout_names_its_limit_and_kernel` | `TimeoutExpired`가 그대로 나와 기대한 `RuntimeFailure` 실패 | `120 s`, kernel 진단, stderr와 원인 보존 |
| `test_a_cached_floor_rejection_cannot_accept_a_lone_failed_probe` | 기존 미달 뒤 lone probe 실패를 무시하고 링크 선택 | 이전 미달 사유로 거부하고 링크 종료 |

첫 선택 실행: `uv run --offline pytest -q tests/test_transport.py tests/test_providers.py -k 'floor or colab_exec_reply_timeout or colab_exec_command_timeout'`, 코드 변경 전 `11 failed, 5 passed, 258 deselected`. Cache 재연결 실패 경우를 추가한 Red 실행은 `2 failed, 50 deselected`. 마지막 lone probe 회귀는 구현 전 `1 failed, 52 deselected`였다.

초기 수정 후 같은 선택 실행은 `17 passed, 258 deselected`였다. Lone probe 수정까지 포함한 두 파일 전체 실행은 `276 passed in 39.04s`였다. `uv run --offline ruff check letify tests`와 `git diff --check`도 통과했다.

## Full suite

전체 suite는 `uv run --offline pytest -q`로 한 번 실행했다. 결과는 **1432 passed, 20 skipped, 0 failed**, 소요 시간은 **399.17 s (0:06:39)**다. 프로세스 종료 코드는 `0`이다. 사용자가 제공한 main checkout 집계는 `1555 passed, 7 skipped, 0 failed`이며, 이 worktree의 새 환경에서는 skip 수가 더 많다. 환경이 다른 두 집계를 같은 테스트 구성으로 해석하지 않는다.

## 커밋

- `9a56c9c` `Docs: Specify a shared link floor and refuse unmeasured escapes`
- `0495085` `Test: Reproduce floor escapes and Colab reply timeout diagnostics`
- `e9fcf01` `Test: Prevent a cached floor rejection from accepting a failed probe`
- `072dae3` `Docs: Align troubleshooting with the shared throughput floor`
- `bd63e36` `Fix: Refuse floor escapes and explain Colab exec reply failures`
