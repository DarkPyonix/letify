# Kaggle 가속기 수정 보고서

`TPU_V3_8` 요청 누락, `T4` 카드 수, 가속기 명세를 수정했습니다. 실제 Kaggle 계정에 요청하지 않았습니다. API 지원 여부는 사용자가 제공한 `2026-10-08` 실측 결과를 사용했습니다. 테스트는 `tests/conftest.py`의 `FakeKaggleCloud`와 `FakeKaggleCLI`로 실행했습니다.

| 결함 | 수정 위치 | 변경 내용 |
|---|---|---|
| TPU가 CPU 세션으로 시작됨 | `letify/providers/kaggle.py:1056` | GPU는 기존 API 이름 매핑을 사용하고, TPU는 `Instance.tpu`를 전달합니다. `TPU_V3_8` 인스턴스의 `gpu=None`을 유지하면서 `CommitAndRun.compute.accelerator`에 `TPU_V3_8`을 요청합니다. CPU는 가속기 필드가 없는 요청을 유지합니다. |
| T4가 카드 1장으로 선언됨 | `letify/providers/kaggle.py:73`, `letify/providers/kaggle.py:978` | `GPUS["T4"]["devices"]=2`를 기존 `Instance.devices`에 전달합니다. `P100`은 `devices=1`이고, 두 GPU의 `vram_gb=16`은 카드당 메모리입니다. |
| 명세가 카드 수와 API 지원 상태를 설명하지 않음 | `docs/SPEC.md:121`, `docs/SPEC.md:122` | `T4` 2장, `P100` 1장, TPU의 실제 요청 이름을 명시했습니다. `P100`은 API에서 계속 제공되지만 web UI 가속기 메뉴에는 없다고 기록했습니다. |

`T4` 이름은 기존 선언을 유지하기 위해 바꾸지 않았습니다. 카드 수는 인스턴스 속성인 `devices`로 표현합니다. 새 카드 수를 연결하면서 기본 inventory가 항상 `count=1`을 만드는 문제도 확인했습니다. `docs/SPEC.md:123`에 기본 inventory가 발견된 인스턴스의 카드 수를 따른다고 명시하고, `letify/providers/base.py:111`에서 `Devices(..., count=instance.devices)`로 수정했습니다. 명시적인 계정 inventory 설정은 그대로 따릅니다. `tests/test_kaggle.py:306`은 카드 수뿐 아니라 첫 예약 성공, 추가 예약 거부, 해제 후 재예약까지 검증합니다.

TPU 지원 범위는 `docs/SPEC.md:122`에 제한했습니다. `host="remote"`는 함수 전체를 runtime으로 보내므로 TPU 사용은 사용자 코드의 책임입니다. 사용자는 자신의 lock file에 `torch-xla` 또는 JAX를 넣어야 합니다. letify는 해당 라이브러리를 설치하거나 검사하지 않으며, TPU를 요청하고 보고하는 범위까지만 담당합니다. Kaggle은 TPU를 포함한 모든 인스턴스에서 `host="local"`을 거부합니다.

기존 테스트가 TPU 결함을 놓친 이유는 `tests/test_kaggle.py:554`의 `session_channel`이 runtime에 `name`만 넣고 `instance`를 전달하지 않았기 때문입니다. `tests/conftest.py:1558`의 fake는 `CommitAndRun` 본문의 가속기와 관계없이 실행 ID를 반환했습니다. 채널과 worker가 정상 동작하는 것만으로는 TPU 요청을 증명할 수 없었습니다. 새 테스트 `tests/test_kaggle.py:566`은 실제 인스턴스를 runtime에 넣고 `cloud_calls`에서 `CommitAndRun` 본문을 읽어 TPU, T4, P100, CPU 요청을 각각 확인합니다. 기존 assertion은 삭제하거나 약화하지 않았고 sleep도 추가하지 않았습니다.

실패 후 성공 증거:

| 단계 | 명령 | 결과 |
|---|---|---|
| 구현 수정 전 | `uv run --offline pytest -q tests/test_kaggle.py -k 'declares_its_card_count or requests_its_declared_accelerator'` | `2 failed, 4 passed, 84 deselected`. T4는 `1 == 2`에서 실패했고 TPU는 `compute["accelerator"]`의 `KeyError`로 실패했습니다. |
| 가속기 구현 수정 후 | 동일 명령 | `6 passed, 84 deselected` |
| inventory 수정 전 | `uv run --offline pytest -q tests/test_kaggle.py -k declares_its_card_count` | `1 failed, 1 passed, 88 deselected`. T4의 기본 inventory가 `count=1`이어서 실패했습니다. |
| inventory 수정 후 | 첫 명령 재실행 | `6 passed, 84 deselected` |
| 최종 provider 검증 | `uv run --offline pytest -q tests/test_kaggle.py tests/test_providers.py` | `313 passed, 0 skipped, 0 failed in 56.75s` |

전체 suite는 `uv run --offline pytest -q`로 1회 실행했으며 `1452 passed, 20 skipped, 0 failed in 401.86s (0:06:41)`로 종료했습니다. 로그는 `/tmp/wt-kaccel-full-suite.log`에 있습니다. 사용자가 제공한 기본 checkout 집계는 `1579 passed, 7 skipped, 0 failed`입니다. 이번 실행은 worktree의 `.venv`를 사용했고, 여기에는 `torch`, `IPython`, `numpy`가 없습니다. `tests/test_device.py:19`는 `torch`가 없으면 모듈 전체를 수집에서 제외하고, `tests/test_frames.py:87` 등의 tensor 테스트와 `tests/test_notebook.py:21`의 IPython 테스트도 건너뜁니다. 따라서 이번 집계를 기본 checkout과 동일한 검증 범위로 해석하면 안 됩니다.

전체 실행을 시작한 뒤 기본 inventory 문제를 발견해 추가 수정했습니다. 전체 실행에서 이미 import된 코드와 수집된 테스트에는 마지막 변경이 모두 반영됐다고 보장할 수 없습니다. 전체 suite는 재실행하지 않았으며, 마지막 변경은 최종 상태에서 실행한 Kaggle 및 공통 provider 테스트의 `313 passed` 결과로 별도 검증했습니다. 실제 TPU 계산과 GPU 카드 인식은 fake 테스트 범위 밖이며, API의 세 가속기 지원은 사용자 실측에 근거합니다.

명세, 테스트, 코드는 각 단계에서 별도로 커밋했습니다.

- `c84bb6b Docs: Specify Kaggle accelerator requests and device counts`
- `97fce69 Test: Pin Kaggle accelerator requests and GPU card counts`
- `64fd45e Fix: Request Kaggle TPUs and declare two T4 devices`
- `6ddb78c Docs: Specify inventory counts for discovered Kaggle shapes`
- `6a87353 Test: Verify two-card Kaggle shapes can be reserved`
- `872a7f6 Fix: Use discovered device counts in default inventory`

`ruff check`와 `git diff --check`는 통과했습니다. push, merge, branch 전환은 하지 않았으며 커밋에 attribution trailer를 넣지 않았습니다.
