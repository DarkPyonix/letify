# Kaggle API 토큰 수정 보고서

확정된 두 결정을 적용했습니다. 기존 `~/.letify/accounts/<alias>/access_token`을 그대로 사용하며, 파일 경로 대신 파일에 담긴 토큰 값을 CLI 자식 프로세스의 `KAGGLE_API_TOKEN`에 전달합니다. 같은 worktree와 `fix/kaggle-api-token-form` 브랜치를 유지했습니다.

## 제거한 동작

JSON 자격 증명을 생성하거나 읽는 코드, `write_kaggle_token`, `kaggle_config_path`와 그 export, `KAGGLE_CONFIG_DIR` 처리, `KAGGLE_USERNAME` 및 `KAGGLE_KEY` 인증 처리를 제거했습니다. 이전 수정에서 만든 `api_token` 파일 경로도 제거하고 기존 `access_token` 이름으로 통일했습니다. 사용자 이름과 토큰 입력의 prompt 상수도 각각 `KAGGLE_OWNER_PROMPT`, `KAGGLE_TOKEN_PROMPT`로 이름을 바꿨습니다.

`letify/`, `docs/SPEC.md`, `README.md`, `docs/locales/README_ko.md`에서 다음 검색은 일치 항목 0개입니다.

```sh
rg -n 'kaggle.json|kaggle_config_path|KAGGLE_CONFIG_DIR|KAGGLE_USERNAME|KAGGLE_KEY' letify docs/SPEC.md README.md docs/locales/README_ko.md
```

저장소 전체에서는 `tests/test_kaggle.py`의 `kaggle.json` 미생성 assertion과 `KAGGLE_CONFIG_DIR` 부재 assertion만 의도적으로 남겼습니다. 이 보고서에는 제거 대상과 검색 증거를 설명하기 위한 참조가 남습니다. 기존 assertion은 삭제하거나 약화하지 않았습니다.

## 최종 자격 증명 구조

| 파일 | 내용과 용도 |
|---|---|
| `~/.letify/accounts/<alias>/access_token` | `KGAT_`를 포함한 API 토큰 원문, CLI 인증 |
| 같은 디렉터리의 `cookie` | 대화형 세션 시작과 종료 및 Jupyter proxy 접근 |
| 같은 디렉터리의 `username` | `kernels delete -y <owner>/<slug>`의 owner |

토큰을 담는 새 파일은 만들지 않습니다. 기존 37바이트 형태의 파일만으로 quota 명령이 인증되는 테스트를 추가했습니다. 로그인도 `--key`가 생략되면 기존 `access_token`을 재사용합니다. 기존 계정에 owner를 추가할 때는 `letify login kaggle <alias> --replace --username <owner> --cookie <cookie>`를 사용하며, 토큰을 다시 붙여 넣을 필요가 없습니다.

POSIX에서는 디렉터리 권한 `0700`, 파일 권한 `0600`을 유지합니다. 토큰은 설정 파일이나 출력에 기록하지 않습니다. 로그인은 kaggle.com에 표시된 원문을 요구하며, `KGAT_` 접두사와 공백 없는 비어 있지 않은 본문을 검사합니다. 잘못된 값을 정리하거나 재포맷하지 않고 쿠키 검증과 저장 전에 거부합니다. 측정된 본문 길이 32는 보편적인 길이 제약으로 확대하지 않았습니다.

## CLI 전달과 owner

quota와 노트북 삭제 모두 `kaggle_cli_environment`를 사용합니다. 해당 계정의 `access_token` 내용을 읽어 `KAGGLE_API_TOKEN` 값으로 전달하고, 상위 환경의 같은 변수는 덮어씁니다. 토큰이나 토큰 파일 경로를 CLI 인자에 넣지 않습니다. 계정별 토큰 격리 assertion도 유지했습니다.

Kaggle CLI 2.2.4에는 작동하는 파일 기반 토큰 전달 형태가 없습니다. 토큰을 환경에서도 감추려던 의도를 충족할 수 없으므로, 자식 프로세스 환경을 가능한 가장 좁은 전달 범위로 선택했습니다. 이는 프로세스 인자 목록보다 노출 범위가 작습니다. 이 이유를 명세에 기록했습니다.

owner는 인증 조회에서 가져오지 않습니다. 기존 `--username` 또는 사용자 이름 prompt로 입력받아 `username`에 저장하고 삭제 주소에만 사용합니다. 쿠키 검증의 `users.UsersService/GetCurrentUser` 응답에서 현재 코드가 확인하는 값은 표시 이름이므로, 이를 owner로 오인하지 않고 추가 인증 조회도 피하는 선택입니다. owner가 없는 계정의 삭제는 기존처럼 경고를 남기는 최선 노력 처리이며, quota 인증은 owner 없이 기존 토큰만으로 작동합니다.

## quota와 확인된 사실

`_parse_cli_quota`는 JSON list에서 `resource`, `used`, `remaining`, `total`, `refreshAt`을 읽습니다. `h`로 끝나는 유한한 음이 아닌 시간 값을 변환하며, GPU의 사용량, 잔여량, 한도는 응답 값을 직접 사용합니다. 한도를 고정하지 않고 측정된 계정의 GPU 60시간과 TPU 20시간 응답을 테스트합니다. `remaining`도 차감으로 추측하지 않습니다.

`refreshAt`의 ISO 형식을 검사하며, 측정된 값에는 시간대가 없어 `resets_at`은 `None`으로 둡니다. 잘못된 JSON, 예상과 다른 스키마, 잘못된 시간 값 또는 CLI 실패는 quota를 알 수 없는 상태로 보고합니다. 예외를 올리거나 쿠키 조회로 대체하지 않습니다.

`FakeKaggleCLI`는 `KAGGLE_API_TOKEN`이 없는 환경의 호출을 거부하고, quota에는 실제 list 스키마를 반환합니다. `CreateKernelWithSettings`의 slug 반환과 쿠키로 만든 대화형 노트북에 대한 `kaggle kernels delete -y <owner>/<slug>` 삭제는 확인된 사실로 명세에 기록했습니다. 두 사실에 대한 추가 live 검증 필요 문구는 없습니다. 이번 작업에서 실계정 네트워크 요청은 수행하지 않았습니다.

## 실패 후 통과 증거

명세를 먼저 수정하고 `Docs:`로 커밋했습니다. 이후 기존 파일 이름을 요구하는 테스트와 재입력 없이 토큰을 재사용하는 테스트를 작성하고 실행했습니다.

- 이번 수정 전 Kaggle 테스트: `24 failed, 46 passed in 19.37s`.
- 이번 수정 후 같은 테스트: `70 passed in 18.61s`.
- 이전 quota 수정의 별도 실패 증거: 실제 list 파서와 토큰 저장 테스트 `2 failed in 0.19s`. 파서가 실제 스키마에 `None`을 반환한 실패를 수정했습니다.
- 변경 코드와 테스트의 `ruff check`: `All checks passed!`.

파일 권한, 토큰 원문, 출력과 인자 비노출, 계정 격리, owner를 포함한 삭제 주소, 잘못된 토큰 거부, 실제 quota 파싱 및 알 수 없는 quota assertion의 강도를 유지했습니다. sleep을 추가하거나 실행하지 않았습니다.

## 전체 suite와 커밋

전체 suite는 기존 오프라인 실행 환경을 사용해 이번 작업에서 한 번 실행했습니다. 저장소 밖 `/tmp/ktoken-offline-bin/uv`는 기존 로컬 cache를 테스트용 cache에 복사하고 실제 uv를 실행하는 wrapper입니다. 설치된 Python 3.13.15의 bin 경로를 PATH에 넣었습니다. 테스트나 assertion을 환경 문제에 맞춰 변경하지 않았습니다.

```sh
PATH=/tmp/ktoken-offline-bin:/home/brew/.local/share/uv/python/cpython-3.13.15-linux-x86_64-gnu/bin:$PATH UV_OFFLINE=true /tmp/ktoken-offline-bin/uv run pytest -q -p no:cacheprovider
```

정확한 전체 1회 실행 결과는 `1404 passed, 20 skipped in 209.34s (0:03:29)`이며 실패는 0개입니다. 전체 suite를 다시 실행하거나 별도 재검증 결과와 합산하지 않았습니다.

추가 명세 커밋은 `2aee0f2 Docs: Reuse the existing Kaggle access token file`, 테스트 커밋은 `ca8d09d Test: Preserve September Kaggle credentials without repasting`입니다. 남은 구현과 이 보고서는 `Fix: Reuse stored Kaggle access tokens for CLI authentication`으로 커밋합니다. push, develop merge, branch 전환은 하지 않았으며 attribution trailer를 추가하지 않았습니다.
