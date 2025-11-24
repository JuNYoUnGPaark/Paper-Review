## 왜 HAR은 Encoder-Only인가?

HAR에서는... Window -> Activity Label 1개 예측 

즉, 'many-to-one` 분류 문제임. 
- input: (T, C)
- output: y_label

구조적으로 BERT-style encoder + classifier head
- **BERT**: 입력 문장 전체 encoding -> [CLS] Token으로 문장 Label 예측
- **HAR-Transformer**: 입력 윈도우 전체 encoding -> pooled vector로 activity label prediction
  - 여기서 pooled vector란? 어느 Window 전체를 요약한 벡터 
    ```python
    x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
    ```
    - `(B, hidden_dim, T)` -> `(B, hidden_dim)`으로 윈도우 전체를 요약한 벡터
