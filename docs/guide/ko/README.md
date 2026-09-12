<div align="center">

# 🧭 letify 가이드

**작업별 안내서입니다. 지금 하려는 일에 맞는 문서를 고르세요.**

[English guides](../README.md) · [프로젝트로 돌아가기](../../locales/README_ko.md)

</div>

---

## 📚 가이드 목록

| | 가이드 | 이럴 때 읽으세요 |
|---|---|---|
| 1️⃣ | **[시작하기](01-시작하기.md)** | 계정이 있고 10분 안에 뭔가 돌려보고 싶을 때 |
| 2️⃣ | **[프로바이더와 계정](02-프로바이더.md)** | Colab, 연구실 서버, Modal, 엘리스를 추가하거나 계정이 여러 개일 때 |
| 3️⃣ | **[실행 방식 고르기](03-실행-방식.md)** | 루프를 보낼지 CUDA 호출을 중계할지, 계산식과 함께 |
| 4️⃣ | **[환경과 데이터](04-환경과-데이터.md)** | 세션 시작이 느리고 캐시로 해결하고 싶을 때 |
| 5️⃣ | **[스윕과 병렬 실행](05-스윕.md)** | 여러 설정을 동시에 돌리고 싶을 때 |
| 6️⃣ | **[비용 관리](06-비용.md)** | 자기 돈으로 내고 있어서 놀랄 일이 없기를 바랄 때 |
| 7️⃣ | **[문제 해결](07-문제-해결.md)** | 뭔가 실패했고 정확한 원인을 알고 싶을 때 |

---

## 🗺️ 급하다면

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_a

@let.function(gpu=colab.G4)
def train(lr, bs):
    ...
    return {"loss": loss}

with let.run():
    print(train(lr=1e-4, bs=32))
```

다른 걸 읽기 전에 세 가지만 알면 됩니다.

1. **함수를 호출하면 실행됩니다.** `.remote()` 같은 건 없습니다. 동기냐 비동기냐는 `def`로 썼는지 `async def`로 썼는지가 결정합니다.
2. **`with let.run():`이 돈이 시작되고 끝나는 지점입니다.** 그 밖에서 호출하면 예외가 납니다.
3. **letify는 조용히 느린 쪽으로 넘어가지 않습니다.** 쓸 수 없는 방식을 요청하면 이유를 설명하는 예외를 받습니다.

---

## 🧩 가이드가 아닌 참조 문서

| | |
|---|---|
| [PROJECT.md](../../../PROJECT.md) | 전체 기능과 API |
| [docs/SPEC.md](../../SPEC.md) | 현재 설계, 결정 단위로 |
| [docs/COMPONENT.md](../../COMPONENT.md) | 클래스와 용어 |
| [docs/NETWORK.md](../../NETWORK.md) | 전송 경로와 측정된 지연 |
| [docs/INTENT.md](../../INTENT.md) | 목표, 주장, 열린 질문 |
