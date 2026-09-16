<div align="center">

# ✨ letify

### 선언이 인프라가 됩니다.

**함수에 필요한 것만 적으면, 감당 가능한 GPU에서 돌아갑니다.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-black)](../../LICENSE)
[![Providers](https://img.shields.io/badge/providers-Colab%20%7C%20Modal%20%7C%20SSH%20%7C%20Local-6C5CE7)](#-프로바이더)
[![Pure Python](https://img.shields.io/badge/pure-python-2ECC71)](../../pyproject.toml)

[빠른 시작](#-빠른-시작) · [왜 선언인가](#-접속하는-대신-선언하는-이유) · [프로바이더](#-프로바이더) · [문서](../) · [English](../../README.md)

</div>

---

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_pro_plus

@let.function(device=colab.G4, host=letify.remote)
def train(lr, bs):
    import torch
    loss = ...                          # 학습 루프
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

세션을 만들 필요도, 환경을 설치할 필요도, 파일을 올릴 필요도 없습니다. 노트북을 닫아도 GPU가 혼자 돌아가지 않습니다. 🎉

---

## 💡 접속하는 대신 선언하는 이유

GPU를 빌린다는 것은 보통 그 GPU에게 찾아가는 일입니다. 노트북이나 SSH 세션을 열고, 거기에 환경을
다시 만들고, 데이터를 복사해 넣고, 그 머신이 살아 있는 동안 남의 컴퓨터 안에서 작업합니다. 학습은
그중 작은 부분입니다. 나머지는 결과를 만들지 않는 인프라 작업입니다.

선언은 이것을 뒤집습니다. 함수가 무엇을 필요로 하고 어디에 속하는지만 말하면, 세션과 환경과 전송과
정리는 알아서 준비됩니다.

### 🧠 작업하던 환경을 떠나지 않습니다

에디터, 디버거, 메모, 데이터, git 이력이 전부 있던 자리에 그대로 있습니다. 선언은 함수 하나를 카드로
보내고 결과를 받아옵니다. 원격 GPU는 옮겨가서 사는 장소가 아니라 함수 하나의 속성입니다.

이 연속성이 핵심입니다. 파이썬 버전도 다르고 작업 디렉터리도 없는 브라우저 탭 속의 또 다른 나로
갈라지지 않습니다. 맞춰줄 것도 없고, 세션이 끊기기 전에 되가져올 것도 없습니다.

```python
@let.function(device=colab.G4, host=letify.remote)
def train(lr, bs):
    ...

train(lr=1e-4, bs=32)     # 방금까지 편집하던 그 파일
```

### ⚡ 인프라를 켜고 끄는 단계가 없습니다

켤 것이 없고, 실수로 남는 것도 없습니다. 호출이 필요한 세션을 시작하고 일이 끝나면 끝내므로, GPU
끄는 것을 잊는 실수는 할 수가 없습니다. 패키지 설치도 그렇습니다. 환경은 이미 가지고 있는
`uv.lock`에서 나오고, 캐시되어 두 번째 세션은 그 비용을 내지 않습니다.

여기서 없어지는 실패는 비싼 쪽입니다. 잊은 인스턴스는 밤새 과금되고, 20분 들여 준비한 세션은 그
안의 모든 것과 함께 사라집니다.

### 📈 하나의 서버에 묶이지 않습니다

선언은 머신이 아니라 가속기의 모양을 지목합니다. 그래서 같은 코드가 설정 한 줄만 바꾸면 Colab
런타임에서도, SSH로 붙는 연구실 머신에서도, Elice allocation에서도, 이 노트북에서도 돕니다. 계정
하나가 소진되면 아무것도 고치지 않고 계정을 늘립니다.

수평 확장도 같은 방식입니다. 용량은 프로바이더 항목이 가졌다고 선언한 것이고, 동시에 부른 호출들은 내가 쓸 수
있는 모든 카드로 퍼집니다. 계정을 넘어서, 머신을 넘어서.

```toml
[colab_pro.devices]
G4 = { count = 2 }            # 이 계정은 세션 두 개

[lab_a100.devices]
A100 = { indices = "0-3" }    # 공용 머신의 네 장이 우리 것
```

카드가 여섯 장이면 설정 여섯 개가 동시에 돕니다. 그 하드웨어가 같은 하드웨어일 필요는 없습니다.
자원이 늘어나도 선언은 바뀌지 않습니다.

### 💸 그래서 감당 가능해지는 것

같은 카드인데 어느 문으로 빌리느냐에 따라 가격이 크게 다릅니다.

| 같은 카드, 다른 문 | 시간당 |
|---|---|
| 🥇 Colab 크레딧 | **약 975원** |
| 💸 Modal | 약 4,070원 |

같은 실리콘에 네 배 차이입니다. 다만 싼 쪽은 노트북입니다. 영구 디스크가 없고, 언제든 끊기고, 그때
비어 있는 GPU를 받습니다. 위의 것들이 바로 그 문을 쓸 만하게 만드는 것들이고, 그래서 가장 싼 선택이
가장 불편한 선택이 아니게 됩니다.

<table>
<tr><th width="50%">😖 GPU에게 찾아가기</th><th width="50%">😌 선언하기</th></tr>
<tr valign="top"><td>

```python
# 브라우저 열고, GPU 고르고, 비어 있기를 기대
!pip install -q torch transformers  # 4분
!gdown ...                           # 18분
!git clone https://github.com/me/repo
%cd repo
# ... 이제야 작업 시작
# ... 세션 끊김, 처음부터 다시
```

</td><td>

```python
@let.function(device=colab.G4, host=letify.remote, volumes=[cache])
def train(lr, bs):
    ...

train(lr=1e-4, bs=32)
```

</td></tr>
</table>

---

## 📦 설치

```bash
uv add letify
```

모든 프로바이더에 이것 하나면 됩니다. letify는 cloudpickle과 blake3만 설치합니다. 머신에 [uv](https://docs.astral.sh/uv/)가 있어야 합니다. Colab CLI와 Modal 클라이언트 같은 프로바이더 도구는 여러분의 `.venv`가 아니라 uv로 별도 환경에서 실행되기 때문입니다.

휠에는 [letify-core](../../letify-core/)가 들어 있습니다. `host=letify.local`에서 CUDA 드라이버를 대신하는 Rust 구성 요소입니다. 휠은 Linux x86_64와 aarch64, Windows x86_64와 arm64, macOS arm64와 x86_64용으로 빌드됩니다. 다른 플랫폼에서는 letify-core가 없는 소스 배포판이 설치되고 `host=letify.local`은 거부됩니다.

---

## 🚀 빠른 시작

### 1. 계정을 한 번 선언합니다

계정마다 프로젝트 디렉터리에서 `letify login`을 한 번 실행합니다.

```bash
letify login colab colab_pro_plus
letify login shell lab_a100
```

명령 하나가 파일 두 개를 씁니다. 계정은 `~/.letify/config.toml`에 들어갑니다. 머신에 속하는 정보이고, 저장소에는 들어가지 않습니다.

```toml
[colab_pro_plus]
kind = "colab"
account = "you@example.com"

[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true
```

프로젝트의 `.letify/config.toml`에는 alias만 들어갑니다. 이 줄이 있어야 그 계정을 이 프로젝트에서 쓸 수 있습니다.

```toml
[colab_pro_plus]

[lab_a100]
```

홈 파일에서 `global = true`인 계정과 `local`은 프로젝트에 적지 않아도 됩니다.

> 🔐 비밀은 `config.toml`에 넣지 않습니다. `access_token` 같은 필드는 `access_token_env`가 가리키는 환경 변수, 또는 `letify login`이 소유자 전용 권한으로 쓰는 `~/.letify/accounts/<alias>/access_token` 파일에서 읽습니다.

### 2. 뭐가 있는지 봅니다

```bash
$ letify providers
colab_pro_plus       colab      ephemeral
lab_a100             shell      persistent
local                local      persistent
```

### 3. 선언하고 실행합니다

```python
import letify

let = letify.Launcher()
env = letify.Env()                      # uv.lock을 읽습니다
colab = let.providers.colab_pro_plus

@let.function(device=colab.G4, host=letify.remote, env=env)
def train(lr, bs):
    loss = ...                          # 학습 루프
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

프로그램 전체가 이게 끝입니다. 🍰

---

## 🧭 선언이 말하는 것

단어 두 개이고, 둘 다 방식을 말하지 않습니다.

```python
@let.function(
    device=colab.G4,       # 가속기가 있는 곳, 프로바이더와 계정까지 포함
    host=letify.remote,    # 호스트 코드가 도는 곳
)
```

**`device`** 는 프로바이더, 계정, 가속기를 한 값에 담습니다. 셋은 하나의 결정이기 때문입니다. 코어 수와 메모리는 프로바이더가 등록한 사양에 딸려 오므로 따로 요청할 것이 없습니다.

**`host`** 는 CUDA에서 CPU 쪽을 가리키는 말입니다. `letify.local`은 Python과 라이브러리를 이 프로세스에 두고 CUDA 호출만 중계합니다. `letify.remote`는 함수를 장치가 있는 머신으로 보냅니다. 문자열 `"local"`, `"remote"`도 같은 값입니다.

`host`를 생략하면 `letify.local`이 됩니다. 장치가 멀면 CUDA 호출을 매번 중계하는 것이 느리므로, 호출할 때 예상 효율이 담긴 경고를 띄운 뒤 실행합니다. 원격 GPU라면 보통 `host=letify.remote`가 맞습니다.

세션이 얼마나 오래 사는지는 선언 인자가 아닙니다. 호출 하나가 끝나면 세션도 끝납니다. `with let.keep_alive():` 블록 안에서는 세션이 유지되므로, 따로따로 부르는 호출들이 매번 세션 시작 비용을 내지 않습니다.

<details>
<summary><b>📐 어떤 host를 고를지, 계산식과 함께</b></summary>

| | 📦 `host=letify.remote` | 🔌 `host=letify.local` |
|---|---|---|
| 무엇이 오가나 | 루프 전체가 한 번 | 모든 CUDA 호출이 매번 |
| 비용 | 전송 한 번 | 호스트 동기화마다 왕복 한 번 |
| 파인튜닝, 왕복 150 ms | **약 99%** | 기본 53%, 조정 시 약 96% |
| 디코딩, 왕복 150 ms | **초당 수백 토큰** | 초당 2~7토큰 |

직접 실행 대비 효율은 `T / (T + k × RTT)`입니다. `T`는 스텝당 GPU 시간, `k`는 그 스텝에서 호스트가 장치의 값을 읽어오는 횟수입니다.

Hugging Face 학습 스텝은 기본 설정에서 `k ≈ 3`입니다. Trainer의 NaN 검사, SDPA 어텐션 마스크 검사, 로깅입니다. NVFP4 마이크로스텝 0.5초에 왕복 150 ms면 53%입니다. NaN 검사를 끄고, 고정 길이 패킹으로 마스크 검사를 없애고, 그래디언트 누적 경계에서만 로깅하면 약 96%가 됩니다.

직관과 어긋나는 결과가 하나 있습니다. **GPU가 빠를수록 중계가 불리해집니다.** `T`는 줄어드는데 `RTT`는 그대로이기 때문입니다. 같은 스텝이 L4에서는 1.8초라 80%가 나옵니다.

디코딩은 계속 나쁜 경우입니다. 처리량이 초당 `1000 / (k × RTT)` 토큰 근처에 묶여서 카드 성능이 의미가 없어집니다. 이때는 `generate` 호출 전체를 보내세요.

`k`는 `torch.cuda.set_sync_debug_mode("warn")`으로, 왕복 시간은 `letify probe`로 직접 재보면 됩니다. [docs/NETWORK.md](../NETWORK.md)를 보세요.

</details>

> ⚠️ letify는 **절대** 몰래 모드를 바꾸지 않습니다. `host`는 선언한 그대로 동작합니다. 프로바이더가 해줄 수 없는 것을 요청하면 이유가 담긴 예외를 받습니다. 느린 것을 요청하면 숫자가 담긴 경고를 받고, 그대로 실행됩니다. 선택은 사용자의 몫이기 때문입니다. letify가 스스로 고르는 것은 머신까지 가는 네트워크 경로 하나이고, 동작하는 경로 중 가장 빠른 것을 고릅니다.

---

## 🌍 프로바이더

```
Provider
├── 💻 Local      내 컴퓨터              persistent
├── ☁️  Modal      서버리스 GPU           persistent
└── 🐚 Shell      원격 머신 전반         기본 ephemeral
    ├── 📓 Colab   공식 CLI 경유
    ├── 🕳️  Tunnel  NAT 뒤의 머신
    └── 🇰🇷 Elice   엘리스 클라우드, API로 할당
```

| | 저장소 | 어디에 맞나 |
|---|---|---|
| 💻 `Local` | persistent | 내 GPU, 그리고 나머지 전부의 테스트 |
| ☁️ `Modal` | persistent | 프로덕션 서빙, 재현 가능한 이미지 |
| 📓 `Colab` | ephemeral | 저렴한 배치 작업, `G4`에서 NVFP4 |
| 🐚 `Shell` | 덮어쓰기 가능 | 연구실과 학교 서버 |
| 🕳️ `Tunnel` | 덮어쓰기 가능 | 포트를 열 수 없는 NAT 뒤의 머신 |
| 🇰🇷 `Elice` | persistent | 한국 GPU 클라우드, 초 단위 과금 |

**letify가 가장 빠른 연결 방법을 찾습니다.** `Shell` 계열 머신에 대해 letify는 여러 연결 방법을 동시에 시도하고, 성공한 것 중 가장 빠른 방법을 씁니다.

1. 머신 주소로 바로 SSH
2. TCP 홀펀칭, 양쪽이 모두 NAT 뒤에 있을 때
3. [Tailcat](https://github.com/tailscale/tailcat)으로 UDP 홀펀칭한 뒤 그 위로 SSH
4. 프로바이더 자체 경로, 예를 들어 `colab exec`와 Colab 파일 API

번호가 낮은 방법이 이깁니다. 다만 연결된 방법 중 가장 빠른 것보다 훨씬 느리면 탈락합니다. 이긴 방법은 계정과 네트워크별로 기억해 두고, 다음 연결에서 먼저 시도합니다. NAT 뒤에 있는 일반 머신은 letify를 설치한 뒤 그 머신에서 `letify client shell connect`를 한 번 실행해야 letify가 접속할 수 있습니다. 그 머신의 SSH 서버가 Docker의 `-p 30501:8022`처럼 다른 포트로 외부에도 열려 있다면 `letify client shell connect --ssh-port 8022 --public-address <host> --public-port 30501`로 실행합니다. 그러면 주소로 바로 하는 SSH는 30501 포트로 접속하고, 홀펀칭과 Tailcat은 계속 8022를 씁니다. 이미 있는 계정에는 `letify login tunnel <alias> --address <host> --public-port 30501`로 같은 값을 설정합니다. Colab과 Elice는 이 단계가 아예 필요 없습니다. 프로바이더 계층이 각 서비스의 API로 머신을 만들고 여는 과정이 `letify client shell connect`를 대신하기 때문입니다. Modal은 자체 API로 연결하므로 여기에 해당하지 않습니다.

**여러 계정을 정식으로 지원합니다.** 설정 항목 하나가 계정 하나이고, 같은 종류를 여러 개 둘 수 있습니다. Colab 계정이 두 개면 동시 세션도 두 배가 됩니다.

```python
a = let.providers.colab_pro_plus
b = let.providers.colab_pro

@let.function(device=a.G4, host=letify.remote)
def train(lr): ...

@let.function(device=b.L4, host=letify.remote)      # 다른 계정, 같은 프로그램
def evaluate(ckpt): ...
```

**아예 고르지 않아도 됩니다.**

```python
@let.function(device=let.providers.any.A100, host=letify.remote)   # A100이 있는 첫 프로바이더
def train(lr): ...
```

---

## ⚡ 여러 호출을 한꺼번에

함수를 `async def`로 선언하면 호출이 코루틴을 돌려줍니다. `with let.keep_alive():` 안에서 `asyncio.gather`로 원하는 만큼 함께 돌리면, 호출마다 세션을 새로 열지 않고 세션을 나눠 씁니다.

```python
import asyncio

@let.function(device=colab.G4, host=letify.remote)
async def train(lr, bs):
    loss = ...                          # 학습 루프
    return {"loss": loss}

async def main():
    with let.keep_alive():
        return await asyncio.gather(*(
            train(lr=lr, bs=bs) for lr in (1e-4, 3e-4, 1e-3) for bs in (16, 32)
        ))

results = asyncio.run(main())           # 결과 6개, 요청한 순서대로
```

호출들은 계정이 선언한 카드 수만큼 동시에 돌고, 카드가 모두 바쁘면 하나가 빌 때까지 기다립니다. 동시 실행 수를 선언에서 정하지 않습니다.

> 🧵 **동기와 비동기는 호출이 아니라 `def` 자리에서 선언합니다.** 평범한 `def`는 블로킹이고 값을 돌려줍니다. `async def`는 코루틴을 돌려주니 `await`, `asyncio.gather`, `asyncio.as_completed`가 평소와 똑같이 동작합니다. letify가 자체 future 타입을 만들지 않고, 외울 `.remote()`나 `.spawn()`, `.map()`도 없습니다.

---

## 💾 실제로 도움이 되는 캐시

**볼륨**은 내용 주소 블롭 저장소입니다. 내용을 해시로 이름 붙이고, 바뀌는 이름은 별도의 작은 공간에 둡니다. Git의 객체와 ref와 같은 구조입니다.

```python
project = colab.volume("my-project")    # 이 프로젝트에 필요한 것의 복사본, 버킷에 보관

@let.function(device=colab.G4, host=letify.remote, volumes=[project])
def train(lr): ...
```

런타임은 볼륨을 여러분의 컴퓨터를 거치지 않고 버킷에서 바로 받습니다. 이때 로컬 로그인에서 잠시 빌린 짧은 수명의 토큰을 쓰므로, 원격에는 인증 정보가 남지 않습니다.

학습 데이터는 선언할 필요가 없습니다. `pathlib.Path`를 인자로 넘기거나 전역 변수에서 읽으면, letify가 그 경로의 파일을 내용 주소 블롭으로 보냅니다. 함수 본문은 같은 구조를 가진 런타임 쪽 경로를 받습니다. persistent 머신은 블롭을 자기 디스크에 두므로 두 번째 세션의 업로드는 0바이트입니다. `bucket = "<이름>"`을 설정한 ephemeral 계정은 파일마다 버킷에 한 번만 올리고, 이후 런타임은 모두 버킷에서 받습니다. 런타임에 둔 사본은 예산 안에서 보관하며(기본 50 GiB, 계정의 `data_cache_gib`로 변경), `letify cache`로 보거나 비울 수 있습니다. 명세상으로는 첫 번째 물결에 해당하는 데이터가 놓이는 즉시 호출이 시작되고 나머지는 호출에서 뽑아낸 순서대로 뒤에서 전송되지만, 그 부분은 아직 구현되지 않았습니다. 지금은 없는 파일을 전부 올린 뒤에 호출이 시작됩니다.

결과도 같은 방식으로 돌아옵니다. 아직 없는 `Path`나 디렉터리는 출력 위치이기도 합니다. 함수 본문이 그곳에 만들거나 바꾼 파일은 호출이 끝나면 로컬 경로로 복사되고, 로컬에 이미 같은 내용이 있는 파일은 보내지 않습니다. 그래서 `Path("runs/exp1")` 아래에 저장한 체크포인트와 로그는 호출이 끝나면 프로젝트에 들어와 있습니다.

```python
DATA = Path("data/imagenet-subset")

@let.function(device=lab.A100, host=letify.remote)
def train(lr):
    for file in DATA.iterdir(): ...   # 이미 런타임 디스크에 있음
```

이 구조를 고른 이유입니다.

| | 🐌 양방향 파일 동기화 | ⚡ 내용 주소 방식 |
|---|---|---|
| 동시에 쓰는 경우 | 마지막이 이기고 나머지는 사라짐 | 구조적으로 충돌 불가 |
| 이미 보냈는지 확인 | 크기와 수정 시각 비교 | 해시를 가진 것이 **곧 증거** |
| 작은 파일 5만 개 | 왕복 5만 번 | 묶음 하나, 전송 한 번 |

가장 아픈 지점인 세션 시작에서 효과가 납니다.

| 20GB 모델 캐시 받기 | 시간 |
|---|---|
| 🐢 연구실 서버에서 100Mbps로 | 약 27분 |
| 🚶 Hugging Face 허브에서 | 3~5분 |
| 🚀 런타임 옆 버킷에서 | **40~60초** |

이 시간은 전부 GPU 크레딧으로 청구됩니다. 그래서 볼륨을 붙인 ephemeral 프로바이더가 persistent처럼 동작합니다.

---

## 🔗 제자리에 남는 값

세션은 살아 있는 프로세스 하나이므로, 값이 그 안에 남아 있을 수 있습니다. `letify.session_cache`는 세션이 처음 요청할 때 값을 만들고, 그 세션의 이후 호출에는 같은 객체를 돌려줍니다.

```python
def load_model():
    return ...                   # 14 GB, 세션마다 한 번만 로드

@let.function(device=colab.G4, host=letify.remote)
def evaluate(batch):
    model = letify.session_cache("model", load_model)
    return model(batch)

with let.keep_alive():               # 세션이 호출보다 오래 삽니다
    evaluate(batch=first)            # 모델을 로드
    evaluate(batch=second)           # 그대로 재사용
```

값은 그 세션이 끝날 때까지 삽니다. 세션마다 따로 가지므로 호출이 어느 세션에 떨어지든 상관없고, 카드 두 장에서 동시에 부른 호출은 각자 한 벌씩 로드합니다. 런타임 밖에서, 예를 들어 함수 본문을 로컬에서 테스트할 때는 같은 동작을 하는 평범한 프로세스 내 캐시입니다.

스크립트의 전역 딕셔너리로는 이렇게 되지 않습니다. 함수가 호출될 때마다 스크립트 전역 값의 복사본과 함께 런타임으로 보내지므로, 그런 캐시는 매번 빈 상태로 시작합니다.

큰 인자는 내용 주소로 다룹니다. 같은 텐서를 열 번 넘겨도 네트워크는 한 번만 건넙니다. 런타임에게 다이제스트로 이미 가지고 있는지 묻기 때문입니다.

---

## 💸 청구서가 폭주할 수 없습니다

손으로 내릴 것이 없습니다.

**호출이 자기 세션을 끝냅니다.** 이게 기본입니다.

**`with let.keep_alive():`가 선택 사항입니다.** 이어지는 호출들이 매번 세션 시작 비용을 내야 하는
경우를 위한 것입니다. 블록 안에서 `asyncio.gather` 등으로 동시에 부른 호출들은 계정이 선언한 카드 수만큼 함께 돌고, 카드가 모두 바쁘면 하나가 빌 때까지 기다립니다. 블록을 나가면 쉬고 있는 세션이 모두 끝납니다. 타이머로 끝내는 것은 없습니다.

**할당할 수 없는 장치는 예외를 냅니다.** 유지 중인 유휴 세션이나 다른 프로세스가 잡고 있는 카드, 또는
계정이 선언한 것보다 많은 카드를 요청하면 기다리지 않고 바로 `letify.InsufficientDevices`를 냅니다.

**리스가 안전장치입니다.** 세션이 기한을 들고 있고 이 프로세스가 계속 갱신합니다. 강제 종료된
프로세스는 아무에게도 아무 말을 할 수 없으므로, 워커가 스스로 나가면서 카드 점유를 풉니다. 유예
시간이 넉넉해서 회선이 불안정한 것만으로 학습이 죽지는 않습니다. 과금까지 멈추는지는 프로바이더가
무엇에 대해 요금을 매기는지에 달려 있고, [비용](../guide/ko/06-비용.md)에 어느 쪽인지 적혀 있습니다.

> 🚫 **분리 실행은 일부러 넣지 않았습니다.** 분리해서 돌리다가 원격이 선점되면 결과까지 잃습니다.
> 대신 로컬 프로세스가 소유자로 남고, 저장소의 체크포인트가 지속성을 담당합니다.

---

## 🧪 GPU도 목 객체도 없이 테스트

`local` 프로바이더는 원격 런타임과 **똑같이** 직렬화된 호출을 똑같은 드라이버 스크립트로 실행합니다. 테스트가 실제 경로를 지나갑니다.

```python
def test_train_returns_a_loss():
    let = letify.Launcher(home=False)

    @let.function(device=let.providers.local.CPU, host=letify.remote)
    def train(lr):
        return {"loss": 1.0 / lr}

    assert train(lr=2.0)["loss"] == 0.5
```

---

## 🛠️ CLI

```bash
letify login shell lab # 계정을 등록하고, 이 저장소에서 참조
letify logout lab     # 이 머신에서 계정 제거
letify client shell connect  # NAT 뒤 원격 머신에서 실행해, letify가 접속할 수 있게 함
letify setup tailcat  # 배포처에서 tailcat 또는 eci를 미리 설치
letify providers      # 선언된 프로바이더, 저장소 수명, 기본 배치
letify devices           # 각자 제공하는 GPU
letify status         # 지금 돌고 있는 것
letify usage          # 계정마다 남은 사용량
letify utilization    # 인스턴스별 GPU가 얼마나 바쁜가
letify check lab      # 이 머신이 응답하나?
letify probe lab      # 호출 중계를 쓸 만큼 가까운가?
```

`usage`, `utilization`, `status`에 `--json`을 붙이면 프로그램이 읽을 수 있는 출력이 나옵니다. [letify-ext/](../../letify-ext/)의 VS Code 확장이 이 출력으로 상태 표시줄에 사용량과 GPU 활동을 보여 줍니다.

---

## 📚 문서

| | |
|---|---|
| 🧪 [examples/](../../examples/) | 동작하는 시나리오, 빌린 카드에서의 LoRA 학습부터 |
| 📖 [PROJECT.md](../../PROJECT.md) | 전체 기능과 API |
| 🎯 [docs/INTENT.md](../INTENT.md) | 목표, 주장, 제약, 열린 질문 |
| 📐 [docs/SPEC.md](../SPEC.md) | 현재 설계, 결정 단위로 |
| 🧩 [docs/COMPONENT.md](../COMPONENT.md) | 모든 클래스와 용어 |
| 🌐 [docs/NETWORK.md](../NETWORK.md) | 전송 경로, 지연 측정, 터널 선택 |
| 🧭 [docs/guide/ko/](../guide/ko/) | 작업별 가이드 |
| 🇺🇸 [README.md](../../README.md) | English |

---

## 🚧 상태

알파이고, 그 점을 숨기지 않습니다. 지금 동작하는 것입니다.

✅ 선언, 동기와 비동기, 풀링, 세션 수명과 리스
✅ 상주 세션: 세션 캐시가 호출 사이에 값을 유지하고, 큰 인자는 한 번만 전송됩니다
✅ 내용 주소 저장소, 설정과 비밀 관리
✅ `Local`과 `Colab` 프로바이더
✅ `letify-core`를 실제 GPU에서 확인: 에이전트가 드라이버를 열고, 로컬 드라이버가 할당과 양방향 복사를 중계하며 바이트가 일치합니다

아직 안 된 것입니다.

🚧 `letify-driver`는 PyTorch 프로세스가 시작해서 커널 하나를 돌리는 데 필요한 진입점까지만 덮습니다. 그 밖의 것은 자기 이름을 출력하고 `CUDA_ERROR_NOT_SUPPORTED`를 반환하므로, 실제 실행이 다음에 만들어야 할 목록을 알려줍니다
🚧 `Modal`과 `Elice`는 공개된 인터페이스대로 작성했지만 실제 서비스에서 돌려보지 않았습니다
🚧 통합 메모리는 중계가 불가능하므로, 페이지드 옵티마이저는 `host=letify.remote`가 필요합니다

전체 목록은 [docs/SPEC.md](../SPEC.md) 끝에 있습니다.

---

## 🤝 기여

스펙 우선, 테스트 우선입니다. [docs/SPEC.md](../SPEC.md)를 먼저 정하고, 실패하는 테스트를 쓰고, 그다음 코드를 씁니다. 성능 작업은 [ResearchTree](https://darkpyonix.github.io/researchtree/)를 씁니다. 브랜치 하나가 실험 하나이고, 풀 리퀘스트 하나가 그 실험 노트입니다. 올리기 전에 [CLAUDE.md](../../CLAUDE.md)를 읽어주세요.

---

<div align="center">

**Apache 2.0 라이선스.** 자기 GPU 비용을 직접 내는 사람들을 위해 만들었습니다. 🔬

</div>
