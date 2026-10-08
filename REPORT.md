# Colab 수정 보고

원격 tailcat 설치, 로컬 tailcat 실행 경로, daemon 시작 확인, wake 재시도와 직렬화를 수정했습니다. 별도로 pool의 사용 중 runtime 종료 경쟁 조건과 등록 key 변경 문제를 수정했습니다. 환경 archive 실험은 구현하지 않았습니다.

사용자가 확인한 `10:31` 이후 `200 OK` ping과 생성 약 `22분` 뒤 VM 소실은 그대로 받아들입니다. `404`가 letify의 정리보다 먼저였으므로 Colab이 먼저 runtime을 회수했고 letify가 뒤늦게 발견했습니다. 회수 원인은 미확인입니다. 이번 daemon 수정이 그 회수를 막는다는 주장은 하지 않습니다.

## PROBLEM 1: 원격 tailcat 부재

원인은 `letify/transport/nat.py:378`의 `tailcat serve` 실행 전에 Colab runtime에 binary를 준비하는 경로가 없었다는 것입니다. 로컬 binary 경로는 원격 VM에 존재하지 않습니다.

`letify/providers/colab.py:328`의 provider 준비 단계에서 연결 race 전에 설치하도록 결정했습니다. race 안에서 설치하면 설치 오류가 한 strategy의 실패로 처리되어 다른 링크로 진행할 수 있기 때문입니다. `letify/install.py:611`은 기존 `TAILCAT_VERSION`, `TAILCAT_RELEASE`, `TAILCAT_SHA256`을 원격 source에 넣습니다. `letify/install.py:635`에서 runtime의 architecture를 선택하고, `letify/install.py:650`에서 release archive를 다운로드한 후 pinned SHA-256을 검사합니다. 안전한 regular file만 추출합니다. 원격 cache를 재사용할 때도 binary digest를 검사합니다.

검증된 절대 경로를 `letify/transport/rendezvous.py:90`의 request에 전달합니다. 설치 오류는 race 진입 전에 `RuntimeFailure`로 전달하며, CLI가 remote error를 출력만 한 경우에도 출력 내용을 error에 보존합니다. 실제 Colab에서 tailcat이 probe를 이기거나 `2`부터 `5 MiB/s`보다 빨라지는지는 측정하지 않았습니다.

## PROBLEM 2: uv sync 이후 로컬 tailcat 경로

`letify/install.py:219`는 project `.venv/bin/tailcat`에 cache binary의 hard link 또는 copy를 만들었습니다. provider는 그 project 경로를 실행 경로로 선택했습니다. `.venv` 항목이 제거되면 이미 선택한 경로가 무효가 됩니다.

원본은 수정 전에도 `~/.letify/tools/tailcat/0.6.0/tailcat`에 보관되어 있었습니다. 원본까지 `.venv` 안에만 있었다는 설명은 이 checkout과 맞지 않습니다. `letify/install.py:208`과 `letify/providers/shell.py:150`을 바꾸어 letify가 관리하는 binary는 원본 cache 경로로 실행합니다. project link가 제거되어도 실행 경로가 유지됩니다. 사용자가 직접 설치한 tool이나 `tailcat_binary` override는 기존 선택을 유지합니다.

테스트는 project link 제거를 fake pruning으로 재현합니다. 실제 사용자 project에서 `uv sync`가 삭제한 파일의 종류와 uv 버전별 pruning 동작까지 재현한 것은 아닙니다.

## PROBLEM 3: daemon 시작과 버전

확인된 letify 결함은 수정 전 `letify/providers/colab.py:350`의 `create_session`이 `colab new` 성공만 확인하고 daemon의 시작을 확인하지 않았다는 것입니다. CLI 자체의 child stderr도 letify가 보존하지 않았습니다. 따라서 `session_created`만 기록되어도 letify는 정상 시작으로 취급했습니다.

관측된 daemon 무기록 종료의 직접 원인은 확정하지 못했습니다. 공식 `0.6.0` wheel의 `colab_cli/commands/session.py:386`을 확인하면 이미 `start_new_session=True`로 detach하고 있습니다. `stdout`와 `stderr`는 `DEVNULL`로 버립니다. 로컬 cache에서 `sys.executable -m colab_cli.cli --help`는 offline 상태에서 exit `0`이었습니다. 그러므로 uv 부모 종료가 반드시 child를 죽인다는 설명을 확인된 원인으로 적을 수 없습니다. 실제 실패 시 interpreter 경로, 초기 traceback, session state와 child PID가 필요합니다.

`letify/providers/colab.py:388`에서 letify의 interpreter로 standalone `colab_keepalive.py`를 직접 시작합니다. 이 daemon은 짧게 실행되는 Colab CLI의 child가 아닙니다. POSIX에서는 별도 session으로 detach합니다. `letify/providers/colab_keepalive.py:103`에서 `source=letify`인 `keep_alive_started`를 history에 기록하고 flush와 fsync를 마친 후 readiness를 보냅니다. provider는 `10 s` 안에 readiness를 받아야 성공합니다. 실패하면 child를 종료하고 할당한 Colab session도 정리합니다. stderr는 account history 옆 `<session>.letify.log`에 남깁니다. 기존 CLI가 자체 daemon을 만드는 release에서는 그 daemon과 별도로 letify supervisor가 동작합니다.

버전 사실도 바로잡습니다. `letify/tools.py:42`의 package는 `google-colab-cli`이며 `==0.6.0` pin은 없습니다. `python=3.13`과 `jupyter-kernel-client<1`만 지정되어 있습니다. 이번 환경의 uv cache에는 `0.6.0`이 있었습니다. 공식 `0.7.4` wheel은 `jupyter-kernel-client==0.9.0`을 요구하고 기존 `spawn_keep_alive` 구현도 없습니다. 단순 upgrade는 daemon 시작 보장에 대한 답이 아닙니다. [공식 0.6.0 배포](https://pypi.org/project/google-colab-cli/0.6.0/), [공식 0.7.4 배포](https://pypi.org/project/google-colab-cli/0.7.4/).

`0.7.4`를 대상으로 `new`, `exec`, session state, file API 호환성을 검증한 뒤 명시적으로 버전을 고정하는 후속 작업은 권장합니다. 이 branch에서는 package나 dependency pin을 바꾸지 않았습니다. daemon의 시작 확인과 VM 회수 원인 조사는 서로 별개의 문제입니다.

## PROBLEM 4: wake 실패와 account 충돌

현재 checkout에는 `5분` 간격의 wake loop가 없었습니다. 수정 전 `letify/providers/colab.py:274`의 `_exec`에도 account file lock이 없었습니다. 따라서 보고된 예전 `5분` retry의 정확한 구현 위치는 이 코드에서 확인하지 못했습니다. 기존 `letify/runtime/lease.py`의 `300 s`는 worker lease grace이며 kernel wake 간격이 아닙니다.

새 supervisor의 `letify/providers/colab_keepalive.py:55`는 성공 후 `60 s`, 실패 후 `5 s`에 다음 wake를 시도합니다. 이는 명령 종료 이후의 대기 시간이며 lock 대기와 최대 `120 s`인 명령 실행 시간은 별도입니다. `letify/providers/colab_keepalive.py:30`의 account file lock은 전체 wake 명령을 감쌉니다. `letify/providers/colab.py:278`의 일반 provider `_exec`도 같은 lock을 사용합니다. 서로 다른 process라도 같은 CLI account directory를 사용하면 직렬화됩니다.

실패 로그에는 session, command, exit status 또는 exception, 전체 stderr와 retry 간격을 한 줄로 기록합니다. stderr의 newline은 escape합니다. 실패한 wake 자체는 runtime을 폐기하거나 새 VM을 만들지 않습니다.

## PROBLEM 5: 살아 있는 session 종료

코드에서 재현한 결함은 `RuntimePool.release_idle`의 후보 수집과 `discard` 사이 경쟁입니다. idle 후보가 다른 호출에 의해 다시 획득되어 `busy=True`가 되어도 수정 전 `discard`는 무조건 종료했습니다. `letify/runtime/pool.py:217`에서 pool guard 안에 `busy`와 hold 재검사를 넣고, 같은 guard 안에서 등록 목록에서 제거합니다. `release`와 마지막 hold 종료 모두 이 조건을 사용합니다. deterministic 테스트는 실제 worker를 사용하며 후보 선택 직후 다시 획득하는 순서를 강제합니다.

또한 `Env.key`는 `uv.lock`, `pyproject.toml`, `.python-version` 내용을 읽어 계산합니다. 수정 전 `Runtime.key`도 매번 이를 다시 계산하여 파일 변경 후 등록 bucket에서 제거되지 않을 수 있었습니다. `letify/runtime/pool.py:253`과 `letify/runtime/session.py:104`에서 획득 당시 등록 key를 보존합니다. 다른 key로 새 획득을 요청하면 `letify/runtime/pool.py:97`에서 기존 key와 요청 key를 기록하며 기존 runtime을 폐기하지 않습니다.

`letify/runtime/session.py:166`은 종료 로그에 session, provider, key, busy 상태, reason과 상세 error를 남깁니다. `call_complete`, `last_hold_closed`, `infrastructure_failure`, `boot_failure`, `interpreter_mismatch`, `explicit_discard`, `explicit_shutdown`, `pool_shutdown`을 구분합니다. infrastructure 실패 상세에는 exception과 그 error가 가진 stderr도 들어갑니다.

이 경쟁 조건이 사용자의 `10:45` 사건을 일으켰는지는 확정할 수 없습니다. 다음 관측으로 후보를 구분할 수 있습니다.

| 후보 경로 | 코드와 판단 | 다음에 필요한 관측 |
|---|---|---|
| 잘못된 idle clock 또는 timer reap | 현재 pool에는 idle timeout이나 timer reap이 없습니다. `idle_for`는 monotonic 값을 사용하며 상태 표시용입니다. | `last_hold_closed` 또는 `call_complete` reason과 block 종료 시점 |
| idle 후보의 재획득 | 이번 테스트로 사용 중 worker의 종료를 재현했고 수정했습니다. | 종료 전 busy 상태와 동시에 시작된 호출 |
| project 변경으로 key 불일치 | 동일 instance에 다른 environment key로 요청하면 추가 runtime이 필요할 수 있습니다. 기존 runtime은 유지합니다. | `pool key mismatch`의 두 key와 project 파일 변경 시점 |
| health read의 거짓 실패 | `pool.live`는 `letify/runtime/pool.py:302`에서 등록 목록만 반환하며 health를 읽지 않습니다. `Colab.sessions()`도 pool reaping에 사용되지 않습니다. | 폐기 reason과 실제 failing command의 stderr |
| channel 또는 infrastructure 실패 | `letify/declare/function.py:156`은 `RuntimeFailure` 또는 `ProtocolError`이면 provider diagnosis 뒤 runtime을 폐기하고 retry합니다. live VM에 대한 transient 통신 실패도 후보입니다. | `infrastructure_failure` 상세, worker exit code, 같은 endpoint의 동시 상태 |
| provider lost 분류 | Colab에는 `KaggleSessionEnded` 같은 별도 lost exception이 없습니다. `letify/runtime/channel.py:818`의 closed pipe와 worker exit 등이 `RuntimeLost`를 만듭니다. Kaggle의 `letify/providers/kaggle.py:562`는 실패한 proxy 상태를 여러 원인의 가능성으로 표현합니다. | CLI error와 endpoint 상태를 같은 시각에 비교 |
| lease 만료 | `letify/runtime/lease.py:54`는 renewal 실패 시 loop를 끝냅니다. `letify/protocol/worker.py:1717`은 wall clock deadline을 사용하고 만료 시 worker를 종료합니다. 이번 branch에서 lease 동작은 바꾸지 않았습니다. | 마지막 lease reply, worker 종료 시각, clock 변화, `300 s` grace와의 관계 |
| boot 실패 또는 process 종료 | `letify/providers/base.py:443`의 boot cleanup과 `pool.shutdown`은 의도적으로 할당을 정리합니다. | `boot_failure`, `interpreter_mismatch`, `pool_shutdown` reason |

## PROBLEM 6: VM 교체 비용, 보고만 수행

사용자의 약 `10분`은 `600 s`의 관측값이며 이번 작업의 실측값이 아닙니다. 사용자 workload의 dependency 목록, archive 크기, dataset 크기와 단계별 시간 기록이 없어 정확한 분 단위 분해는 불가능합니다. 코드상 비용 경로는 다음과 같습니다.

| 단계 | 코드 | 반복 비용의 근거 |
|---|---|---|
| VM 생성과 연결 | `letify/providers/colab.py:372`, `letify/providers/shell.py:268` | 새 VM마다 `colab new`, tool 준비, SSH 연결과 probe 수행 |
| 선언 환경 설치 | `letify/runtime/session.py:567`, `letify/runtime/bootstrap.py:197` | archive가 없으면 project 파일 전달, uv 설치, 선언 Python 준비, `uv sync`, 추가 packages와 commands 수행. 새 ephemeral disk에는 기존 `.venv`가 없습니다. |
| interpreter 전환 | `letify/runtime/session.py:622`, `letify/runtime/channel.py:660` | 환경을 만든 뒤 worker를 새 interpreter로 이동하고 hello를 확인합니다. 전환 자체가 dependency 재설치는 아닙니다. `120 s` timeout을 실제 소요 시간으로 해석하면 안 됩니다. |
| data 첫 전달 | `letify/runtime/session.py:278`, `letify/runtime/session.py:326` | 새 VM의 data cache가 비어 있으므로 first wave 전달과 이후 background 전송이 다시 필요합니다. |

전송 비용은 `bytes / throughput`으로 도출할 수 있습니다. 보고된 `2`부터 `5 MiB/s`에서는 `1 GiB`만 옮겨도 약 `205`부터 `512 s`, 즉 `3.4`부터 `8.5분`이 걸립니다. 실제 데이터 크기를 모르므로 이 값을 사건의 `10분`에 그대로 배정할 수는 없습니다. `20분` cycle에서 setup `10분`이면 setup만 `50%`를 차지합니다.

기존 `letify/store/volume.py:154`의 `env/<Env.key>-<platform>` 참조와 `cache_env_from`이 재설치 비용을 줄이는 경로입니다. `letify/runtime/session.py:590`은 ephemeral provider에 연결한 volume에서 archive를 먼저 찾고, 복원 검증에 성공하면 `uv sync`를 건너뜁니다. 최초 sync 후에는 archive를 저장합니다. 기본적으로 volume이 없으면 이 cache 경로도 사용되지 않습니다.

volume을 연결한 최초 archive miss에는 추가 비용도 있습니다. `letify/store/volume.py:171`의 `cache_env_from`은 remote environment를 `pack_dir`로 client에 가져와 backend에 저장합니다. 큰 environment archive라면 이 최초 생성과 업로드가 setup 시간을 더할 수 있습니다. 반복 VM에서는 저장된 archive를 복원하므로 이 단계도 반복하지 않습니다.

Colab의 `store_backend()`는 `letify/providers/colab.py:242`에서 `gcs`를 기본으로 선택합니다. bucket을 사용하는 volume의 `letify/store/volume.py:223`은 remote pull로 archive를 직접 materialize합니다. 환경 재설치가 주된 비용이라면 이 environment archive와 동일 cloud bucket 경로가 대부분의 반복 설치 비용을 없애는 기존 수단입니다. dataset과 model cache도 bucket에서 VM으로 전달하면 느린 로컬 링크 비용을 줄일 수 있습니다. 실제 절감률은 archive 복원 시간과 data 양을 측정해야 합니다. 이번 branch에서는 volume 설정이나 cache 방식 변경을 구현하지 않았습니다. 후속 실험은 `docs/INTENT.md`의 `N3`, `N4`와 전체 wall time을 기준으로 평가해야 합니다.

## Red 다음 Green 검증

spec을 먼저 commit하고 테스트를 작성했습니다. 구현 전 첫 선택 실행은 `11 failed, 371 deselected`, readiness 실패 테스트의 별도 실행은 `1 failed, 213 deselected`였습니다. 구현 후 동일한 `12`개 case를 함께 실행한 결과는 `12 passed, 371 deselected in 2.34s`입니다. 기존 assertion을 삭제하거나 약하게 만들지 않았고 새 sleep을 넣지 않았습니다.

| 테스트 | Red 근거 | Green 확인 |
|---|---|---|
| `test_the_provider_uses_the_durable_tailcat_after_uv_prunes_the_project_link` | cache 대신 `.venv/bin/tailcat` 반환 | cache 절대 경로 선택, project link 제거 후에도 binary와 lookup 유지 |
| `test_colab_verifies_the_release_on_the_runtime_before_installing[False]` | `remote_tailcat_source` 부재 | fake release 설치 후 binary 변조 시 digest 오류 |
| `test_colab_verifies_the_release_on_the_runtime_before_installing[True]` | `remote_tailcat_source` 부재 | 잘못된 pinned digest 거부, binary 미설치 |
| `test_colab_prepares_tailcat_before_the_connection_race` | 준비 명령이 실행되지 않음 | fake exec에 검증 source 전달, rendezvous 절대 경로, 설치 오류 전파 |
| `test_colab_requires_the_daemon_to_acknowledge_startup` | owned daemon 없음 | uv CLI와 별도 interpreter로 detach, readiness 확인, stop 시 child 종료 |
| `test_a_daemon_that_exits_before_ready_fails_creation_and_cleans_up` | readiness 없이 creation 성공 | 빈 readiness 거부, child와 Colab session cleanup |
| `test_failed_wakes_retry_after_five_seconds_and_log_stderr` | supervisor module 없음 | 실패 `5 s`, 성공 `60 s`, command timeout `120 s`, escaped stderr |
| `test_account_wakes_hold_the_same_file_lock_until_the_command_finishes` | supervisor module 없음 | 첫 holder가 lock을 놓기 전 다른 실행은 진입하지 못함 |
| `test_the_detached_daemon_records_startup_before_acknowledging` | supervisor module 없음 | 실제 detached process, history 기록 후 readiness, 별도 session과 종료 확인 |
| `test_idle_release_rechecks_a_session_acquired_after_the_idle_snapshot` | 재획득한 busy runtime까지 종료 | 같은 실제 worker가 pool에 남음 |
| `test_runtime_disposal_names_its_reason_and_the_registered_key` | 폐기 로그 없음 | session, `call_complete`와 key 기록 |
| `test_a_runtime_keeps_its_registered_key_when_project_files_change` | lock 수정 시 registered key 변화 | 등록 key 유지, release 후 pool이 비어 있음 |

## 전체 suite와 한계

전체 suite는 요청한 `uv run pytest -q -p no:cacheprovider`로 `1회` 실행했습니다. 결과는 `1342 passed, 0 failed, 16 skipped in 180.93s`입니다. 사용자가 제공한 develop baseline `1465 passed, 0 failed, 3 skipped`와 같은 환경의 결과는 아닙니다. 이 worktree에서 새로 생성된 `.venv`에 `torch`가 없어 `tests/test_device.py` module이 수집에서 빠지고 `tests/test_frames.py` 일부가 skip되었습니다. project dependency는 변경하지 않았습니다.

추가 검증을 위해 기존 uv cache에서 `torch==2.14.0+cpu`와 `mypy==2.3.1`을 offline 설치했습니다. 전체 suite는 다시 실행하지 않고 영향받는 `tests/test_device.py`, `tests/test_frames.py`, `tests/test_kaggle.py`를 별도로 실행했습니다. 결과는 `192 passed, 0 failed, 0 skipped, 7 warnings in 199.24s`입니다. 경고는 기존 forwarding 테스트의 tensor scalar 변환과 Python의 multi-threaded fork 관련입니다. 이 추가 결과는 일부 기존 통과 case도 포함하므로 전체 suite 결과에 단순 합산하지 않습니다.

테스트는 기존 `tests/conftest.py`의 CLI/process fake, local release server와 실제 local worker를 사용했습니다. 실제 account나 Colab endpoint에는 요청하지 않았습니다. 공식 PyPI metadata와 wheel source만 외부에서 읽어 version 관련 설명을 검증했습니다. 변경 코드와 새 daemon 테스트의 `ruff check`, `git diff --check`도 통과했습니다.

실제 Colab VM에서 release 다운로드와 tailcat server 시작, link 성능, CLI release별 kernel wake 성공, account 동시 실행, VM 회수 정책은 검증하지 않았습니다. daemon 시작 기록이 없었던 직접 원인, `10:45` 폐기의 정확한 원인, setup `600 s`의 단계별 비중도 확정하지 않았습니다. Windows detach, process liveness와 file lock은 이 Linux 환경에서 실행하지 않았습니다.

spec, tests, code를 별도 commit으로 남겼습니다.

| commit | 내용 |
|---|---|
| `7ff3c22` | `Docs: Specify Colab preparation and session supervision` |
| `23ffb50` | `Test: Cover Colab tools daemon startup and runtime disposal` |
| `1e1f849` | `Fix: Prepare Colab tools and supervise session wakes` |

 branch 변경, push와 develop merge는 하지 않았습니다.
