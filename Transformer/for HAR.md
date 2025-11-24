## 왜 HAR은 Encoder-Only인가?

HAR에서는... Window -> Activity Label 1개 예측 

즉, 'many-to-one` 분류 문제임. 
- input: (T, C)
- output: y_label

구조적으로 BERT-style encoder + classifier head
- **BERT**: 입력 문장 전체 encoding -> [CLS] Token으로 문장 Label 예측
- **HAR-Transformer**: 입력 윈도우 전체 encoding -> pooled vector로 activity label prediction
  - 여기서 pooled vector란? 어느 Window 전체를 요약한 벡터 (ModernTCN의 `PAMAP.ipynb`)
    ```python
    x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
    ```
    - `adaptive_avg_pool1d` -> `(B, hidden_dim, T)` -> `(B, hidden_dim)`으로 시간축 T 방향으로 평균을 계산 -> 윈도우 전체를 요약한 벡터 = Pooled Vector

IF Decoder가 필요하다면? 
- 미래 시계열 예측
- 센서 -> 텍스트로 설명 등

## HAR에 Encoder-only Transformer를 적용할 때 데이터의 흐름
1. 입력: `X: (B, T, C)'
2. Input Projection (=Embedding): `h0: (B, T, d_model)`
3. Positional Encoding: 텍스트와 동일 `h0 -> Z0`
4. Self-Attn + FFN 반복: `(B, T, d_model)`
   - Self-Attn: 시간 축 전체를 보며 정보 섞기 -> "걸음 중간중간 발이 땅에 닿는 패턴 / 주기 / 앞뒤 구간의 연결" 등등..
   - FFN: 각 시점의 벡터를 non-linear MLP로 가공하여 비선형으로 강화
5. Window -> Label 하나로 압축 (Pooling + Classifier)
   - 마지막 encoder의 출력 `(B, T, d_model)`에서 window 대표 vector 하나만 뽑기 = GAP / ATttn Pooling 등 
   - GAP or Attn pooling -> Linear -> SoftMax
   - `(B, T, d_model)` -> `(B, d_model)` -> `(B, n_classes)`
  

## CNN/TCN vs. Encoder-only Transformer 
1. CNN/TCN
   - 국소적 커널(3, 5, 7 step)로 움직이는 패턴 감지에 강함
   - 깊게 쌓을수록 넓은 receptive field 확보 가능
   - 시간축 translation invariance로 인한 좋 inductive bias
       - translation invariance: "어딩 있든 동일한 패턴이면 비슷하게 반응하는 성질"
       - inductive bias(귀납적 편향): "모델이 학습 전 이미 가지고 있는 세상을 어떻게 볼지에 대한 선입견"
   - BUT: 시점들간의 상관관계 및 중요도는 kernel에 암묵적으로 들어있는 정도

2. Transformer Encoder
   - Attention score를 통해 시점들 간 서로 얼마나 중요한지를 학습
   - 반복된 패턴이 없거나 장기 의존성이 중요한 HAR에서는 더 유연한 표현 가능
   - BUT: Conv/TCN보다 inductive bias가 약해서 데이터가 적으면 성능 떨어질 수 있음 (대체로 낮음)
