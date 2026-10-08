# Kaggle API 토큰 수정 보고서

`KGAT_` 접두사를 보존한 토큰을 파일에 저장하고, 모든 Kaggle CLI 호출에 `KAGGLE_API_TOKEN` 환경 변수로 전달하도록 수정했습니다. 사용자에게 제공받은 Kaggle CLI 2.2.4 측정 결과만 사용했으며, 계정에 대한 네트워크 요청은 수행하지 않았습니다.

## 제거한 동작

`letify/config/login.py`에서 `KAGGLE_KEY_PREFIX`와 접두사를 제거하는 처리, `write_kaggle_token`의 legacy `kaggle.json` 생성을 제거했습니다. `letify/tools.py`에서 `kaggle_config_path`와 `KAGGLE_CONFIG_DIR` 설정을 제거했습니다. 상위 환경의 `KAGGLE_CONFIG_DIR`도 CLI 자식 환경에서 제외합니다. 사용하는 명령 중 이 설정이 필요한 명령은 없습니다.

`letify/providers/kaggle.py`에서 `name`, `totalTimeAllowed`, `timeUsed`를 추측하던 quota 파서와 초 단위를 시간으로 변환하던 처리를 제거했습니다. quota 스키마의 실계정 확인이 필요하다는 문구를 제거하고, `CreateKernelWithSettings`의 slug 반환과 CLI 삭제가 확인된 사실을 명세에 기록했습니다.

## 최종 자격 증명 구조

계정 디렉터리는 `~/.letify/accounts/<alias>/`이며, POSIX 권한은 디렉터리 `0700`, 파일 `0600`입니다. 기존 `write_secret` 경로를 사용하므로 파일 생성 시점부터 소유자만 접근할 수 있습니다. Windows에서는 기존과 같이 상위 디렉터리의 접근 권한을 상속합니다.

| 파일 | 내용과 용도 |
|---|---|
| `cookie` | 브라우저 쿠키, 대화형 세션 시작과 종료 및 Jupyter proxy 접근 |
| `api_token` | `KGAT_`를 포함한 API 토큰 원문, CLI 인증 |
| `username` | 노트북 삭제 주소에 사용할 owner |

`kernels delete`와 `quota`는 모두 `kaggle_cli_environment`를 통해 해당 계정의 `api_token`을 `KAGGLE_API_TOKEN`으로 받습니다. 상위 환경의 다른 토큰은 해당 계정의 토큰으로 덮어씁니다. 토큰은 subprocess 인자, 설정 파일, 출력에 포함되지 않습니다. 계정 2개의 토큰 격리도 테스트했습니다. 기존 `kaggle.json`만 있는 계정은 `letify login kaggle <alias> --replace`로 원문 토큰과 쿠키를 다시 입력해야 합니다. 기존 로그인 정책에서 `--replace`가 계정의 자격 증명을 갱신하는 옵션입니다.

로그인은 kaggle.com에 표시된 `KGAT_` 포함 토큰을 그대로 붙여 넣도록 안내합니다. 접두사와 공백 없는 비어 있지 않은 본문을 검사하며, 입력을 정리하거나 접두사를 제거하지 않습니다. 측정된 본문 길이 `32`는 모든 토큰의 길이 규칙을 증명하지 않으므로 길이를 고정하지 않았습니다. 잘못된 입력은 쿠키 검증과 파일 저장 전에 거부하고 무엇을 붙여 넣어야 하는지 안내합니다.

## quota 파싱

실제 JSON list의 `resource`, `used`, `remaining`, `total`, `refreshAt` 필드를 읽습니다. `h`로 끝나는 유한한 음이 아닌 시간 값을 변환합니다. GPU의 `used`, `remaining`, `total`은 각각 `Usage.used`, `Usage.remaining`, `Usage.limit`에 직접 대응합니다. `remaining`을 차감으로 추측하지 않고, 주간 한도를 고정하지 않습니다. 제공된 응답은 GPU `60.00h`, TPU `20.00h`입니다. TPU는 `Usage.resources`와 안내 문구에 포함됩니다.

`refreshAt`은 ISO 8601 형태를 검사합니다. 제공된 시각에는 시간대가 없으므로 Unix 시각을 추정하지 않고 `Usage.resets_at`을 `None`으로 둡니다. 잘못된 JSON, 예상과 다른 스키마, 잘못된 시간 값, CLI 실패는 예외 대신 알 수 없는 quota와 이유를 반환합니다. 쿠키 호출로 대체하지 않습니다.

## owner 결정

기존 `--username`과 사용자 이름 입력을 유지했습니다. 인증에는 쓰지 않고 `kernels delete -y <owner>/<slug>`의 주소에만 사용합니다. 추가 인증 조회를 피하고, 쿠키 검증이 반환하는 표시 이름을 owner로 오인하지 않기 위한 선택입니다. 명세에 이유를 기록했고, 저장된 owner가 삭제 인자에 반영되는 것을 검증했습니다.

## 실패 후 통과 증거

`docs/SPEC.md`를 먼저 수정하고 명세를 커밋한 다음, 테스트와 fake를 변경하여 기존 구현의 실패를 확인했습니다.

- 구현 변경 전 관련 테스트: `27 failed, 47 passed in 19.49s`.
- 별도 파서 및 토큰 본문 길이 테스트: `2 failed in 0.19s`. 실제 스키마를 직접 넣은 `_parse_cli_quota`는 `None`을 반환했고, 로그인은 `api_token` 파일을 만들지 않았습니다.
- 구현 변경 후 관련 테스트: `76 passed in 18.84s`.
- 변경 파일의 `ruff check`: `All checks passed!`.

`FakeKaggleCLI`는 `KAGGLE_API_TOKEN` 환경 변수가 없으면 호출을 거부하며, 기본 quota 응답은 제공된 실제 스키마입니다. 접두사 보존, 파일 권한, 환경 전달, 인자 비노출, 잘못된 토큰 거부, 실제 quota 파싱, 알 수 없는 quota 경로를 검증합니다. 측정으로 반박된 기존 기대값은 명세에 맞춰 교체했고, 권한과 비노출 검사의 강도는 유지했습니다. 테스트 통과를 위한 sleep은 추가하지 않았습니다.

## 전체 테스트와 커밋

전체 테스트는 `UV_OFFLINE=true` 환경에서 `uv run pytest -q -p no:cacheprovider` 명령으로 1회 실행했습니다.

정확한 전체 1회 실행 결과는 `10 failed, 1392 passed, 20 skipped in 203.66s (0:03:23)`입니다. 제가 오프라인 실행 환경을 충분히 준비하지 않아 `tests/test_runtime.py`의 `9`개와 `tests/test_store.py`의 `1`개가 실패했습니다. 원인은 테스트용 workspace의 Python 설치 경로와 빈 uv cache였으며, Kaggle 테스트는 통과했습니다.

이미 설치된 `Python 3.13.15`를 `PATH`에 연결하고, `/tmp/ktoken-offline-bin/uv`에서 기존 로컬 cache의 `blake3`, `cloudpickle`, `hatchling`과 빌드 의존성 데이터를 테스트용 cache에 최초 1회 복사한 뒤 실제 uv를 실행했습니다. 준비 도중 누락된 빌드 의존성과 중복 복사를 수정했습니다. 이 준비는 저장소 밖에서 수행했고, 테스트 코드나 assertion을 변경하지 않았습니다. 오프라인 상태를 유지했으므로 다운로드 요청은 없었습니다.

환경 준비 후 실패했던 `10`개만 같은 pytest 옵션으로 다시 실행한 결과는 `10 passed in 11.88s`입니다. 전체 suite를 다시 실행하지 않았습니다. 전체 1회 실행과 실패 항목 재검증을 합치면 고유 테스트 기준 `1402 passed`, `20 skipped`, 미해결 실패 `0`입니다. 이것은 전체 1회 실행의 실패 수를 숨기거나 전체 재실행 결과로 표현한 집계가 아닙니다.

명세, 테스트, 구현은 각각 별도 커밋입니다.

| 커밋 | 내용 |
|---|---|
| `9cb6481` | `Docs: Define verbatim Kaggle tokens and measured quota schema` |
| `2f0b4ab` | `Test: Pin Kaggle token environment and measured quota schema` |
| `4cc22eb` | `Fix: Preserve Kaggle API tokens and parse measured quota hours` |

이 보고서는 별도 `Docs:` 커밋으로 저장합니다. push, merge, branch 전환은 수행하지 않았고, 커밋에 attribution trailer를 추가하지 않았습니다.
