---

- Transformer 대신 쓸 수 있는, 선형 시간 시퀀스 모델 backbone
    - 기본 뼈대는 SSM(State Space Model)
    - 입력에 따라 상태를 선택적으로 업데이트하는 selection mechanism
    - Transformer와 같이 내용-기반 추론이 가능하면서 계산/메모리는 O(L)로 유지하도록 만든 모델

---

### 0. 기본 개념 요약

*Q1. SSM이란?*

- *시퀀스 데이터를 hidden state로 요약해 나가는 점에서 RNN/LSTM과 같은 계열이지만, 핵심은 $h_t = Ah_{t-1}+Bx_t, \ y_t = Ch_t + Dx_t$ 같은 ‘선형 동역학’(state-space) 구조를 사용하는 모델을 말한다.*

*Q2. 동역학이란?*

- *시간에 따라 상태가 어떻게 변하는지를 나타내는 법칙 (ex. 운동 방정식)*
- *“다음 상태는 현재 상태로부터 정해진 규칙에 따라 변한다.”*
- $*h_t = Ah_{t-1}+Bx_t$ ( $h_t$: 현재 상태, $x_t$: 현재 입력, $A, B$: 선형 변환 행렬)*

*Q3. RNN계열과 비교하면?*

- *기존의 RNN/LSTM은 상태를 비선형 함수로 update해서 복잡하고 비효율적*
- *SSM은 상태를 선형식으로 update하여 단순하고 병렬계산이 쉽게 만들어 매우 긴 sequence도 처리가 가능해졌다.*

*Q4. Mamba는?*

- *기존의 SSM은 시간에 따라 A, B가 고정(Linear Time Invarient: LTI)여서 표현력이 떨어짐*
- *Mamba는 $h_t = A_th_{t-1}+B_tx_t$ 로 업그레이드 함. A, B를 입력에 따라 변화시키는(selective)구조로 만들어서 transformer와 같이 내용-기반 추론이 가능해짐.*

---

### 1. LTI SSM이란?

- 가장 기본 SSM 식: $h_t = Ah_{t-1} + Bx_t, \ y=Ch_t$
- ( $h_t$: 현재 상태, $x_t$: 현재 입력, $y_t$: 출력, $A, B, C$: “고정된” 선형 변환 행렬)

1. Linear: 상태 업데이트가 선형식, 곱셈+덧셈만 있음
2. Time-Invarient: A,B,C가 시간에 따라 변하지 않음. 항상 같은 필터로 신호 처리하는 시스템

> **LTI SSM은 1D CNN처럼 볼 수 있다. (수학적 관점)**
> 

$h_t = ah_{t-1}+bx, \ y_t = ch_t$ 일 떄,

- $t=1$:
    - $h_1=ah_0+bx_1$
    - $y_1=ch_1$
    - $y_1=ch_1=c(ah_0+bx_1)=cah_0+cbx_1$
- $t=2:$ $y_2=ch_2=ca^2h_0+cabx_1+cbx_2$
- $t=3$: $y_3=ch_3=ca^3h_0+ca^2bx_1+cabx_2+cbx_3$
- 일반화하면, $y_t=\sum\limits_{l=0}^{t-1} k_lx_{t-l}$
    
    $k_0=cb, \ k_1 = cab, \ k_2=ca^2b, \ ..., k_l=ca^lb$
    
    이 $k_l$들이 바로 “Kernel”이라고 볼 수 있음. 
    
- 1D Convolution의 정의: $y_t=\sum\limits_{l=0}^{L-1} w_lx_{t-l}$
    - 일반 CNN: $w_l$을 직접 파라미터로 두고 학습
    - SSM: A, B, C를 학습하고 그 조합으로부터 $k_l$이 간접적으로 결정됨.

→ 다차원 SSM도 똑같은 구조가 나오고 이땐 그냥 multi-channel 1D conv.와 완전히 같은 꼴이된다. 

### 2. LTI SSM의 한계: “Content 무시” 문제

- 모든 시점, 모든 입력에 대해서 항상 같은 A, B, C를 사용한다는 것. 즉, 내용-기반 선택이 안됨. 표현력의 한계
- 그래서 등장한게 **Selective SSM이고 그 대표 구현이 Mamba.**

### 3. Selective SSM: A, B, C를 “입력에 따라” 바꾸기

- “LTI SSM에서 고정돼 있던 A, B, C를 입력 $x_t$에 따라 동적으로 바꾸자”

$$
h_t = A_th_{t-1}+B_tx_t, \ y_t=C_th_t
$$

- 이렇게 되면,
    - $A_t, B_t, C_t$는 각 Token/Timestep마다 다름.
    - 이 값들은 $x_t$에서 나오는 작은 네트워크를 통해 생성됨.
    - 즉, 입력을 한 번 보고:
        - “얘는 오래 기억/빠르게 잊기/가중치 크게” 등의 **‘선택성’ 부여**

---
