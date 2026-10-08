# Ephemeral 환경 복원 보고서

환경 archive가 volume 선언 없이도 생성되고, 새 VM에서 복원되도록 수정했다. 기존 account의 `bucket`을 재사용하며, bucket이 없으면 로컬 filesystem store를 사용한다. Colab의 one-shot 경로도 bucket에서 VM으로 직접 받는다. 모델 cache 자동화는 spec에 후속 설계로 남겼다. 실제 Colab 준비 시간이 `600 s`에서 `60 s`로 줄었다는 실측 주장은 하지 않는다.

실험 대상은 `docs/INTENT.md`의 setup 비용 일회화 목표와 `N4`이다. `N3`의 session pooling과 별개로 VM 교체 사이의 환경 재사용을 다룬다. 실제 GPU 학습과 direct-run 비교가 없으므로 해당 claim의 성능 판정은 open이다. Colab, Kaggle, Modal, dept_gpu 계정에 연결하지 않았다. 사용한 것은 real Local worker, 기존 `PreparingLocal`, `local_one_shot_runner`, loopback `fake_gcs`이다.

## 기존 준비 시간의 귀속

아래는 `host="remote"`의 준비 단계 분석이다. training step time 표가 아니다. 사용자가 말한 환경 설치와 모델 로딩 합계는 약 `600 s`, 교체 주기는 약 `1320 s`이다. 따라서 이상적인 cycle utilization은 `54.5 percent`이다. 준비 시간이 `60 s`라면 같은 주기에서 `95.5 percent`이며, `85 percent`를 넘으려면 준비 시간이 `198 s`보다 짧아야 한다. 실제 학습 `65 min / 120 min`은 `54.2 percent`로, 사용자가 표현한 약 `50 percent`와 같은 수준이다.

| 단계 | 코드 경로 | 첫 VM의 비용, 반복 새 VM과 비교 |
|---|---|---|
| session 생성 | `letify/providers/base.py:428`, `letify/providers/colab.py:372` | `colab new`와 keep-alive readiness. 둘의 개별 wall time은 `not measured`. timeout은 측정값이 아니다. |
| connection race | `letify/providers/colab.py:453`, `letify/providers/shell.py:288` | 기본 `ssh` 경로에서 pipeline이 선택하고, `exec` 설정은 race를 건너뛴다. 새 session은 다시 연결한다. 개별 wall time은 `not measured`. |
| lease, workspace 준비 | `letify/runtime/session.py:121` | 표준 라이브러리 요청과 디렉터리 준비. 개별 wall time은 `not measured`. |
| 환경 빌드 | `letify/runtime/session.py:601`, `letify/runtime/bootstrap.py:254` | metadata 파일 전송, uv 설치 필요 여부, `uv sync`, 추가 pip packages와 shell commands. Colab 사용자의 개별 wall time은 `not measured`. Modal L4 torch cu128 reference는 sync `100 to 110 s`로, 전체 사용자 준비 `600 s`와 다른 workload이다. |
| interpreter 전환과 검사 | `letify/runtime/session.py:657` | persistent channel은 reexec와 HELLO, one-shot은 child interpreter 설정과 version probe. 개별 wall time은 `not measured`. |
| volume attach | `letify/runtime/session.py:671` | 디렉터리를 만든다. 전체 model cache를 자동 다운로드하거나 mount하지 않는다. 개별 wall time은 `not measured`. |
| 데이터와 weights | `letify/runtime/session.py:287`, `letify/store/pathdata.py:310`, `letify/store/volume.py:239` | call data는 content addressed 전송, volume은 명시적 materialize이다. 사용자 Hugging Face 다운로드, CPU deserialize, GPU copy는 사용자 함수 안에서 발생한다. 각 wall time과 weight size는 `not measured`. |

기존 Colab setup reference는 첫 session `67.05 s`, 새 VM의 두 번째 session `89.7 s`이다. 후자는 전자보다 `22.65 s` 길다. 이 수치는 session setup 전체이며, session 생성 또는 sync만의 시간으로 배분할 수 없다. 기존 link reference는 upload `8.8 MiB/s`, download `10.7 MiB/s`로, download가 upload보다 약 `22 percent` 빠르다. RTT `92.3 ms`는 이 link의 control 왕복 비용이며, bulk throughput을 대체하지 않는다. 자료는 `docs/NETWORK.md`, `docs/SPEC.md:1277` 및 요청에 제공된 merged PR reference이다. 이 branch에서 원격 측정을 추가하지 않았다.

즉, 사용자 전체 `600 s`를 각 단계로 더해서 맞출 근거가 없다. 그중 환경 빌드, model download, model load의 비중을 구분하는 데 필요한 기록은 `not measured`이다. 환경 archive를 연결하는 수정과 VM 회수 원인 조사는 별개이다.

## 원인과 변경

기존 `install_env`는 ephemeral provider에서도 `self.volumes`만 archive 후보로 삼았다. 선언의 기본값은 빈 tuple이다. account의 `bucket`은 `pathdata.data_bucket`에서 project data에만 사용됐고, 환경 volume으로 연결되지 않았다. 따라서 volume이 없는 일반 선언은 archive를 만들지도, 복원하지도 않았다. 명시적 volume을 쓰는 기존 경로는 이미 생성과 복원이 가능했다. `env_root`는 저장 위치를 바꾸며 이 누락의 원인이 아니다. persistent Modal은 별도 disk 환경을 매 session sync하는 기존 정책을 유지한다.

| 변경 | 위치 | 결과 |
|---|---|---|
| 자동 환경 store | `letify/store/volume.py:88` | bucket이 있으면 기존 bucket backend와 prefix를 공유한다. 없으면 `~/.cache/letify/environments/<provider kind>/<account alias>`에 저장한다. bucket 생성이나 새 설정을 요구하지 않는다. |
| archive 후보 선택 | `letify/runtime/session.py:591` | persistent provider는 기존 정책, explicit volumes는 기존 우선순위, 나머지 ephemeral 선언은 자동 cache를 사용한다. |
| 성공 후 publish | `letify/runtime/session.py:146` | interpreter 전환과 version 검사가 성공한 뒤 archive를 만든다. 전환 실패는 blob이나 ref를 publish하지 않는다. |
| managed Python 포함 | `letify/runtime/bootstrap.py:320` | ephemeral sync의 기본 `UV_PYTHON_INSTALL_DIR`을 project의 `.letify-python`으로 옮긴다. uv가 받은 Python과 standard library가 `.venv`와 함께 pack된다. 명시적 `Env.vars`는 우선한다. |
| one-shot 직접 pull | `letify/runtime/session.py:506`, `letify/runtime/bootstrap.py:219`, `letify/store/volume.py:271` | GCS blob을 로컬로 relay하던 제한을 제거한다. VM의 표준 라이브러리 download command가 partial file에 streaming하고 rename과 unpack 후 metadata만 반환한다. |
| model cache 후속 설계 | `docs/SPEC.md:848` | revision별 manifest, shard별 blob, snapshot symlink 검증, partial/lock 제외, read-only tree, `HF_HOME`을 정의한다. |

bucket 없이도 환경 복원은 작동하지만 새 VM으로 bytes를 보내는 비용은 매번 낸다. bucket이 필요한 이유는 이 uplink를 빼는 것이다. 첫 build의 현재 publish는 remote pack, remote-to-client archive 전송, client-to-backend 저장 순서이다. bucket을 써도 첫 publish의 client hop은 남는다. 이후 bucket restore만 직접 pull이다.

model shard를 기존 volume에 넣고 `materialize`하면 origin 다운로드를 피할 수 있다. 그러나 전체 Hugging Face cache는 mutable download state와 snapshot symlink가 섞인 tree이다. shard manifest와 revision 계약을 추가하는 것은 환경 복원보다 큰 별도 변경이므로 spec 후속 설계로 남겼다. 이를 구현해도 새 VM의 CPU deserialize와 GPU weight load는 남는다.

## Local 측정: packing과 unpack 비용

`host="remote"`, real Local framed worker로 random file tree를 pack하고 복원했다. archive 크기는 첫 run과 repeat 모두 `64.020 MiB`, 원본 `64 MiB`보다 약 `0.020 MiB` 크다. 압축이 잘 안 되는 큰 file 하나를 사용했으므로 torch의 수많은 작은 파일을 대표하지 않는다. 아래 unpack 항목은 pipe 전송과 unpack의 합계로, 순수 unpack보다 큰 proxy이다. 두 run을 평균내지 않았다.

| Mode | Run | Pack step wall time | Pipe transfer plus unpack step wall time |
|---|---|---|---|
| `host="remote"`, Local | first | `2.325 s`, repeat보다 `0.056 s` 길다 | `0.204 s`, repeat보다 `0.670 s` 짧다 |
| `host="remote"`, Local | repeat | `2.269 s`, first보다 `2.4 percent` 짧다 | `0.874 s`, first의 약 `4.29 times` |

이 차이는 캐시, scheduling, filesystem 영향을 포함하며 원격 Colab unpack 시간은 `not measured`이다. 아래 크기 외삽은 첫 proxy `0.003182 s/MiB`와 repeat proxy `0.013656 s/MiB`를 각각 사용한다. 두 값의 차이 자체가 외삽 불확실성을 보여준다.

같은 측정 스크립트에서 별도로 file hash body를 Local worker와 직접 실행으로 비교했다. input 배치와 result 반환을 포함하는 whole wall time은 body 비교와 분리했다. direct baseline은 첫 run에 filesystem copy와 result 쓰기, repeat에는 기존 input과 result 쓰기를 포함한다. Local이므로 scp와 원격 네트워크 전송은 없다. archive 준비는 이 hash workload의 입력 작업이 아니므로 위 표에 별도로 측정했다. 이 표는 Colab 학습 효율의 증거가 아니다.

| Mode | Run | Direct whole wall time | letify whole wall time | Direct / letify whole wall efficiency |
|---|---|---|---|---|
| `host="remote"`, Local | first | `0.1183 s`, letify보다 짧다 | `0.3053 s`, direct의 `2.58 times` | `38.74 percent`, direct 기준 `100 percent`보다 낮다 |
| `host="remote"`, Local | repeat | `0.1418 s`, letify보다 짧다 | `0.1715 s`, direct의 `1.21 times` | `82.68 percent`, direct 기준 `100 percent`보다 낮다 |

| Mode | Run | Direct step time only | letify step time only | Direct / letify step efficiency |
|---|---|---|---|---|
| `host="remote"`, Local | first | `0.0730 s`, letify보다 짧다 | `0.1010 s`, direct의 `1.38 times` | `72.25 percent`, direct 기준 `100 percent`보다 낮다 |
| `host="remote"`, Local | repeat | `0.0815 s`, letify보다 짧다 | `0.0975 s`, direct의 `1.20 times` | `83.66 percent`, direct 기준 `100 percent`보다 낮다 |

step 항목은 direct body invocation 또는 client의 동기 call 시작부터 결과까지이다. letify 항목에는 call protocol 비용이 포함되며 setup, input copy, output 쓰기는 제외한다. 실제 GPU step time과 원격 direct-run baseline은 first와 repeat 모두 `not measured`이다. 환경 cache 변경으로 GPU step 자체가 빨라진다고 가정하지 않는다.

## 예상 이득과 손익분기 크기

`E`는 sync로 절약 가능한 시간, `C`는 연결 등 남는 준비, `M`은 모델 다운로드와 로딩, `S`는 compressed archive size in MiB, `P`는 pack time in s, `U`는 unpack time in s이다. `C + E + M = 600 s`를 사용자 reference로 놓는다. 기타 data 전송과 write-back은 이 reference에 따로 기록되지 않았으므로 `not measured`이다. 아래 cycle utilization은 학습 가능한 시간의 비율이며 direct-run 대비 speed efficiency와 다르다.

로컬 store의 첫 실행은 `D_first = 600 + P + S/10.7`이다. repeat 새 VM은 `D_repeat = 600 - E + S/8.8 + U`이다. 첫 실행은 sync를 여전히 지불하고 packing도 추가된다. 반복 실행에는 pack 비용이 없다. bucket의 첫 publish는 여기에 `S/B_upload`가 추가되고, repeat는 `S/B_pull + U`가 restore 비용이다. `B_upload`와 `B_pull`의 단위는 `MiB/s`이며 이번 실험에서는 모두 `not measured`이다. Google 내부 transfer라고 해서 임의로 rate를 넣지 않았다.

| Mode | Run과 가정 | Whole cycle 준비 시간과 비교 | Whole cycle utilization과 비교 |
|---|---|---|---|
| `host="remote"`, local store | first, 작은 archive, first proxy | `608.31 s`, 기존 `600 s`보다 `8.31 s` 길다 | `53.92 percent`, 기존 `54.55 percent`보다 낮다 |
| `host="remote"`, local store | repeat, `E = 105 s`라는 Modal reference 시나리오, first proxy | `502.48 s`, 기존 `600 s`보다 `97.52 s` 짧다 | `61.93 percent`, 기존 `54.55 percent`보다 높다 |
| `host="remote"`, local store | repeat, 같은 `E = 105 s`, repeat proxy | `503.15 s`, 기존 `600 s`보다 `96.85 s` 짧다 | `61.88 percent`, 위 시나리오보다 낮다 |
| `host="remote"`, local store | repeat, `E = 585 s`, `C + M = 15 s`라는 가상 상한 시나리오, first proxy | `22.48 s`, 기존 `600 s`보다 `577.52 s` 짧다 | `98.30 percent`, 기존 `54.55 percent`보다 높다 |

위 시나리오는 측정된 Colab 예측값이 아니다. 작은 archive는 앞서 측정한 `64.020 MiB`이고, 실제 사용자의 환경 archive 크기는 `not measured`이다. 마지막 시나리오에서 남는 준비 `C + M`이 `52.52 s` 이하일 때만 전체 준비 `60 s` 목표에 도달한다. model loading이 이 범위를 넘으면 환경 수정만으로 목표를 달성할 수 없다. `E = 105 s` 시나리오에서는 기존 준비 `10 min`이 약 `8.37 min`으로 줄 뿐이다.

repeat 복원이 이득인 조건은 `S/8.8 + U(S) < E`이다. unpack을 공짜로 놓아도 최대 크기는 `8.8 * E MiB`이다. `E = 105 s` 시나리오의 이 상한은 `924 MiB`, Local first unpack proxy를 포함하면 약 `899 MiB`, repeat proxy로는 약 `825 MiB`이다. proxy를 포함한 두 값은 unpack을 빼고 계산한 상한보다 작다.

첫 build와 새 VM repeat 하나를 합친 비교에서는 `P(S) + S/10.7 + S/8.8 + U(S) < E`가 필요하다. first pack proxy `0.036317 s/MiB`를 적용하면 `E = 105 s`의 손익분기는 약 `426 MiB`, repeat pack/unpack proxy를 별도로 적용하면 약 `410 MiB`이다. 이는 repeat만의 손익분기보다 작다. 일반적으로 repeat가 `K`회이면 `P + S/10.7 < K * (E - S/8.8 - U)`가 amortization 조건이다. bucket 첫 publish에는 추가 upload 항도 넣어야 한다.

PR #12 reference는 archive `2698 MiB`를 packing하는 일이 전체 `620 s` run의 주된 비용이었다는 것이다. 전체 `620 s`를 pack 단독 시간으로 대입할 근거는 없다. 정확한 `P`와 `U`는 `not measured`이다. 위 손익식에서 `P`를 지우면 이 reference의 주된 비용을 누락하게 된다.

이 큰 archive는 기존 link에서 첫 remote-to-client 전송만 `252.15 s`, repeat client-to-VM 전송만 `306.59 s`이다. 각각 `105 s` 환경 build 시나리오보다 길다. first 및 repeat unpack proxy 외삽을 따로 적용하면 repeat restore는 약 `315.18 s` 및 `343.44 s`로, pack을 제외해도 손해이다. `E = 585 s` 가상 시나리오에서도 남은 준비까지 더한 repeat는 약 `330.18 s` 및 `358.44 s`, cycle utilization은 각각 `74.99 percent` 및 `72.85 percent`로, `85 percent` 목표보다 낮다. torch tree의 실제 unpack은 이 외삽과 다를 수 있다.

따라서 archive 연결은 비용 일회화의 필요한 구현이지만 대형 archive가 무조건 sync보다 빠르다는 `N4` 해석은 지지되지 않는다. `N4` 자체는 archive와 file-level synchronization을 비교한다. 여기에는 그 직접 비교가 없으므로 claim을 반증했다고 선언하거나 `docs/INTENT.md`를 수정하지 않는다. 실제 archive 크기, pack/unpack, bucket pull rate, model load를 측정해야 사용자 workload의 gain을 확정할 수 있다.

## 검증과 commit

spec, tests, code를 분리했다. 기존 assertion은 약화하지 않았고 sleep을 추가하지 않았다.

- 자동 archive 관련 red: `uv run pytest -q -p no:cacheprovider tests/test_runtime.py -k 'without_declared_volumes or keeps_managed_python or failed_interpreter_switch'`. 수정 전 `4 failed tests`, 수정 후 같은 선택에서 `4 passed tests`. local/bucket repeat가 `sync`였고, managed Python 경로가 archive 밖이었고, interpreter 전환 실패 전에 blob이 저장되는 이유로 실패했다.
- one-shot direct pull red: `uv run pytest -q -p no:cacheprovider tests/test_store.py -k 'one_shot_runtime_pulls'`. 수정 전 `1 failed test`, 수정 후 `1 passed test`. client backend `get`을 금지한 assertion이 기존 relay를 잡았다. green에서는 unpack한 weights와 downscoped token을 검증했다.
- 기존 runtime/store 회귀 선택: `201 passed tests`, red 선택보다 넓은 실제 worker와 backend 경로를 검증했다. 뒤에 추가한 one-shot test는 별도로 green을 확인했다.
- `ruff` 변경 파일 검사: `0 violations`, 허용 기준 `0 violations`와 일치했다.
- Full suite는 요청한 `uv run pytest -q -p no:cacheprovider`를 한 번 실행했다. 결과는 `1380 passed tests`, 제시된 대략적 baseline `1350 passed tests`보다 `30 tests` 많았다. `20 skipped tests`는 제시된 baseline `16 skipped tests`보다 `4 tests` 많았고, `0 failed tests`는 baseline `0 failed tests`와 일치했다. 이번 branch가 추가한 것은 `5 test cases`이며 새 skip marker는 추가하지 않았다. 나머지 inventory 차이를 이번 변경의 증가분이라고 해석하지 않는다. baseline suite 자체를 별도로 실행하지 않았다.

실제 계정 없이 확인하지 못한 것은 Colab VM 회수 원인, 실제 환경 archive 크기와 압축률, 대형 환경의 원격 pack/unpack wall time, Google 내부 bucket rate, account별 bucket 권한, managed Python을 포함한 실제 Colab image 호환성, 모델 다운로드와 CPU/GPU loading, first/repeat GPU step와 direct-run whole wall baseline이다. 외부 interpreter 위치를 명시하거나 system image 또는 absolute project path가 바뀌면 archive가 호환되지 않을 수 있으며 실행 검증 실패 시 sync한다. local blob store의 용량 제한과 대형 archive의 peak memory도 이번 branch에서 추가하지 않았다.
