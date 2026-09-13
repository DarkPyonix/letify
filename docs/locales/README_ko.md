<div align="center">

# ✨ letify

### 선언이 인프라가 됩니다.

**함수에 필요한 것만 적으면, 감당 가능한 GPU에서 돌아갑니다.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-black)](../../LICENSE)
[![Providers](https://img.shields.io/badge/providers-Colab%20%7C%20Modal%20%7C%20SSH%20%7C%20Local-6C5CE7)](#-프로바이더)
[![Pure Python](https://img.shields.io/badge/pure-python-2ECC71)](../../pyproject.toml)

[빠른 시작](#-빠른-시작) · [왜 만들었나](#-왜-만들었나) · [프로바이더](#-프로바이더) · [스윕](#-스윕) · [문서](../) · [English](../../README.md)

</div>

---

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_a

@let.function(device=colab.G4, host="remote", concurrency=3)
def train(lr, bs):
    import torch
    ...
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

세션을 만들 필요도, 환경을 설치할 필요도, 파일을 올릴 필요도 없습니다. 노트북을 닫아도 GPU가 혼자 돌아가지 않습니다. 🎉

---

## 💡 왜 만들었나

같은 GPU인데 어느 문으로 빌리느냐에 따라 가격이 크게 다릅니다.

| 같은 카드, 다른 문 | 시간당 |
|---|---|
| 🥇 Colab 크레딧 | **약 975원** |
| 💸 Modal | 약 4,070원 |

같은 실리콘에 네 배 차이입니다. 자기 돈으로 내는 사람에게는 실험을 몇 번 돌릴 수 있는지가 여기서 갈립니다.

문제는 싼 쪽이 노트북이라는 점입니다. 영구 디스크가 없고, 언제든 세션이 끊기고, 그때 비어 있는 GPU를 받습니다. 그래서 세션 시간을 패키지 재설치와 가중치 재다운로드, 브라우저 탭 지키기에 씁니다.

**letify는 싼 문이 비싼 문처럼 동작하게 만듭니다.** 선언만 하면 세션, 환경, 캐시, 정리를 알아서 처리합니다.

<table>
<tr><th width="50%">😖 letify 없이</th><th width="50%">😌 letify로</th></tr>
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
@let.function(device=colab.G4, host="remote", volumes=[cache])
def train(lr, bs):
    ...

train(lr=1e-4, bs=32)
```

</td></tr>
</table>

---

## 📦 설치

```bash
uv add letify                 # 코어만, 프로바이더 의존성 없음
uv add "letify[colab]"        # Google Colab
uv add "letify[modal]"        # Modal
uv add "letify[shell]"        # SSH, 터널, 엘리스 클라우드
uv add "letify[gcs]"          # Google Cloud Storage 캐시
uv add "letify[s3]"           # S3 호환 캐시
uv add "letify[all]"          # 전부
```

순수 Python입니다. 컴파일 확장도, 빌드할 휠도, 툴체인도 없습니다. 패키지가 없는 프로바이더는 스스로 사용 불가라고 알리고, 나머지는 그대로 동작합니다.

---

## 🚀 빠른 시작

### 1. 계정을 한 번 선언합니다

계정은 `~/.letify`에 둡니다. 머신에 속하는 정보이고, 저장소에는 들어가지 않습니다.

```toml
[colab_a]
kind = "colab"
account = "you@example.com"

[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true
```

> 🔐 비밀은 참조만 합니다. `access_token_env = "MY_TOKEN"`이나 `access_token_keyring = "service/user"`를 쓰세요.

### 2. 뭐가 있는지 봅니다

```bash
$ letify providers
colab_a   colab   ephemeral   host=remote
lab_a100  shell   persistent  host=remote
local     local   persistent  host=local

$ letify devices
{
  "colab_a":  ["A100", "G4", "H100", "L4", "T4", "v5e1", "v6e1"],
  "lab_a100": ["A100"],
  "local":    ["CPU", "GeForce_RTX_4050"]
}
```

### 3. 선언하고 실행합니다

```python
import letify

let = letify.Launcher()
env = letify.Env()                      # uv.lock을 읽습니다
colab = let.providers.colab_a
cache = colab.volume("hf-cache")        # 세션보다 오래 살아남습니다

@let.function(device=colab.G4, host="remote", env=env, volumes=[cache], concurrency=3)
def train(lr, bs):
    ...
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

프로그램 전체가 이게 끝입니다. 🍰

---

## 🧭 핵심 아이디어 하나

**사용자는 자원을 선언하고, 방식은 letify가 고릅니다.**

원격 GPU를 쓰는 방법이 두 가지인데, 내가 어디 있고 GPU가 어디 있는지에 따라 성능이 크게 갈립니다.

| | 📦 함수 전송 | 🔌 CUDA 호출 중계 |
|---|---|---|
| 무엇이 오가나 | 루프 전체가 한 번 | 모든 CUDA 호출이 매번 |
| 비용 | 전송 한 번 | 동기화마다 왕복 한 번 |
| 파인튜닝, 왕복 150ms | **약 99%** | 50~57% |
| 추론, 왕복 150ms | **초당 수백 토큰** | 초당 2~7토큰 |

둘 중 어느 것도 직접 고르지 않습니다. CPU 쪽 작업이 어디서 도는지만 말하면 됩니다.

```python
@let.function(device=colab.G4, host="remote")                   # 기본값: 루프가 원격에서 돕니다
@let.function(device=lab.A100, host="remote", host="local")      # Python은 여기, CUDA 호출만 저쪽으로
```

그리고 기본값은 프로바이더에서 나옵니다. **저장소가 결정합니다.** 프로바이더의 디스크가 세션보다 오래 살면 데이터가 이미 거기 있으니 루프를 보내는 게 자연스럽습니다. 그렇지 않으면 상태를 로컬에 두는 편이 낫지만, 회선이 감당할 때만 그렇습니다.

> ⚠️ letify는 **절대** 조용히 느린 쪽으로 넘어가지 않습니다. 프로바이더가 해줄 수 없는 것을 요청하면, 네 배 느려진 실행이 아니라 계산식이 담긴 예외를 받습니다.

<details>
<summary><b>📐 계산식이 궁금하다면</b></summary>

원격에서 직접 실행한 경우를 기준으로 한 효율입니다.

```
효율 = T / (T + k × RTT)
```

`T`는 스텝당 GPU 시간, `k`는 그 스텝에서 호스트가 장치의 값을 읽어오는 횟수입니다.

Hugging Face 학습 스텝은 기본 설정에서 `k ≈ 3`입니다. Trainer의 NaN 검사, 어텐션 마스크 검사, 로깅입니다. NVFP4 마이크로스텝 0.5초에 왕복 150ms면 53%가 나옵니다.

직관과 어긋나는 결과가 하나 있습니다. **GPU가 빠를수록 중계가 불리해집니다.** `T`는 줄어드는데 `RTT`는 그대로이기 때문입니다. 같은 스텝이 L4에서는 1.8초라 80%가 나옵니다.

`k`는 `torch.cuda.set_sync_debug_mode("warn")`으로 직접 재보면 됩니다. [docs/NETWORK.md](../NETWORK.md)를 보세요.

</details>

---

## 🌍 프로바이더

```
Provider
├── 💻 Local      내 컴퓨터              persistent
├── ☁️  Modal      서버리스 GPU           persistent
└── 🐚 Shell      SSH로 붙는 모든 머신   기본 ephemeral
    ├── 📓 Colab   공식 CLI 경유
    ├── 🕳️  Tunnel  Tailscale 또는 frp, NAT 뒤
    └── 🇰🇷 Elice   엘리스 클라우드, API로 할당
```

| | 저장소 | 어디에 맞나 |
|---|---|---|
| 💻 `Local` | persistent | 내 GPU, 그리고 나머지 전부의 테스트 |
| ☁️ `Modal` | persistent | 프로덕션 서빙, 재현 가능한 이미지 |
| 📓 `Colab` | ephemeral | 저렴한 배치 작업, 스윕, `G4`에서 NVFP4 |
| 🐚 `Shell` | 덮어쓰기 가능 | 연구실과 학교 서버 |
| 🕳️ `Tunnel` | 덮어쓰기 가능 | 포트를 열 수 없는 NAT 뒤의 머신 |
| 🇰🇷 `Elice` | persistent | 한국 GPU 클라우드, 초 단위 과금 |

**여러 계정을 정식으로 지원합니다.** 설정 항목 하나가 계정 하나이고, 같은 종류를 여러 개 둘 수 있습니다. Colab 계정이 두 개면 동시 세션도 두 배가 됩니다.

```python
a = let.providers.colab_a
b = let.providers.colab_b

@let.function(device=a.G4, host="remote")
def train(lr): ...

@let.function(device=b.L4, host="remote")      # 다른 계정, 같은 프로그램
def evaluate(ckpt): ...
```

**아예 고르지 않아도 됩니다.**

```python
@let.function(device=let.providers.any.A100, host="remote")   # A100이 있는 첫 프로바이더
def train(lr): ...
```

---

## 🔭 스윕

병렬 실행은 `.map()` 호출이 아니라 **선언된 공간**입니다. 스칼라가 올 자리에 공간을 넘기면 그 인자가 변한다는 선언이 됩니다.

```python
space = letify.grid(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32])   # 6개
pairs = letify.zip(lr=[1e-4, 3e-4], bs=[16, 32])          # 2개
both  = letify.grid(lr=[1e-4]) | letify.grid(lr=[1e-3])   # 합집합
```

소비하는 방법은 이미 알고 있는 파이썬 문법입니다. 🐍

```python
@let.function(device=colab.G4, host="remote", concurrency=3)
async def train(lr, bs):
    ...

results = await train(space)              # 입력 순서대로 리스트

async for r in train(space):              # 끝나는 대로 하나씩
    print(r)
```

> 🧵 **동기와 비동기는 호출이 아니라 `def` 자리에서 선언합니다.** 평범한 `def`는 블로킹이고, `async def`는 코루틴을 주니 `await`와 `asyncio.gather`가 평소와 똑같이 동작합니다. letify가 자체 future 타입을 만들지 않고, 외울 `.remote()`나 `.spawn()`, `.map()`도 없습니다.

---

## 💾 실제로 도움이 되는 캐시

**볼륨**은 내용 주소 블롭 저장소입니다. 내용을 해시로 이름 붙이고, 바뀌는 이름은 별도의 작은 공간에 둡니다. Git의 객체와 ref와 같은 구조입니다.

```python
cache = colab.volume("hf-cache")

@let.function(device=colab.G4, host="remote", volumes=[cache])
def train(lr): ...
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

## 💸 청구서가 폭주할 수 없습니다

손으로 내릴 것이 없습니다.

**호출이 자기 세션을 끝냅니다.** 이게 기본이고, 스윕도 호출 하나로 세니 6개 조합이 세션을 한 번
열고 한 번 닫습니다.

**`lifetime="process"`가 선택 사항입니다.** 이어지는 호출들이 매번 세션 시작 비용을 내야 하는
경우를 위한 것이고, 더 쓰이지 않으면 유휴 정리가 가져갑니다.

**리스가 안전장치입니다.** 세션이 기한을 들고 있고 이 프로세스가 계속 갱신합니다. 스크립트를
죽이든, 노트북을 잃든, 커널이 터지든 GPU가 스스로 종료합니다. 유예 시간이 넉넉해서 회선이
불안정한 것만으로 학습이 죽지는 않습니다.

> 🚫 **분리 실행은 일부러 넣지 않았습니다.** 분리해서 돌리다가 원격이 선점되면 결과까지 잃습니다.
> 대신 로컬 프로세스가 소유자로 남고, 저장소의 체크포인트가 지속성을 담당합니다.

---

## 🧪 GPU도 목 객체도 없이 테스트

`local` 프로바이더는 원격 런타임과 **똑같이** 직렬화된 호출을 똑같은 드라이버 스크립트로 실행합니다. 테스트가 실제 경로를 지나갑니다.

```python
def test_train_returns_a_loss():
    let = letify.Launcher(home=False)

    @let.function(device=let.providers.local.CPU, host="remote")
    def train(lr):
        return {"loss": 1.0 / lr}

    assert train(lr=2.0)["loss"] == 0.5
```

---

## 🛠️ CLI

```bash
letify login shell lab # 계정을 등록하고, 이 저장소에서 참조
letify logout lab     # 이 머신에서 계정 제거
letify providers      # 선언된 프로바이더, 저장소 수명, 기본 배치
letify devices           # 각자 제공하는 GPU
letify status         # 지금 돌고 있는 것
letify usage          # 계정마다 남은 사용량
letify utilization    # 인스턴스별 GPU가 얼마나 바쁜가
letify check lab      # 이 머신이 응답하나?
letify probe lab      # 호출 중계를 쓸 만큼 가까운가?
```

---

## 📚 문서

| | |
|---|---|
| 🧪 [examples/](../../examples/) | 동작하는 시나리오, 빌린 카드에서의 LoRA 스윕부터 |
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

✅ 선언, 동기와 비동기, 스윕, 풀링, 스코프와 리스
✅ 호출 프로토콜, 내용 주소 저장소, 설정과 비밀 관리
✅ `Local`과 `Colab` 프로바이더, 실제 코드 경로를 지나는 테스트 31개

아직 안 된 것입니다.

🚧 상주 세션 프로세스가 없어서, `Handle`을 나중 호출에서 해소할 수 없습니다
🚧 CUDA 호출 중계는 아직 가능 여부 검사만 있고 클라이언트가 없습니다
🚧 `Modal`과 `Elice`는 공개된 인터페이스대로 작성했지만 실제 서비스에서 돌려보지 않았습니다

전체 목록은 [docs/SPEC.md](../SPEC.md) 끝에 있습니다.

---

## 🤝 기여

이 저장소의 성능 작업은 [ResearchTree](https://darkpyonix.github.io/researchtree/)를 씁니다. 브랜치 하나가 실험 하나이고, 풀 리퀘스트 하나가 그 실험 노트입니다. 올리기 전에 [CLAUDE.md](../../CLAUDE.md)를 읽어주세요.

---

<div align="center">

**MIT 라이선스.** 자기 GPU 비용을 직접 내는 사람들을 위해 만들었습니다. 🔬

</div>
