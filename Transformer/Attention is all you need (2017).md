![KakaoTalk_20251122_012426717](https://github.com/user-attachments/assets/c728e65f-5229-4656-8f25-a4852b2f0fba)
![KakaoTalk_20251122_012426717_01](https://github.com/user-attachments/assets/2dc9bf6c-588e-48a3-98db-8de157b38f58)
![KakaoTalk_20251122_012426717_02](https://github.com/user-attachments/assets/1447c94d-8d1d-4dc3-8ecc-c5ea09571af4)
![KakaoTalk_20251122_012426717_03](https://github.com/user-attachments/assets/d5060404-e7f7-4b65-b337-353c809f4534)
![KakaoTalk_20251122_012426717_04](https://github.com/user-attachments/assets/8ff49e24-6536-4a2f-8d59-0534e528a0cc)
![KakaoTalk_20251122_012426717_05](https://github.com/user-attachments/assets/f6adefc4-1788-40fb-b5c7-ea0a815a0de0)
![KakaoTalk_20251122_012426717_06](https://github.com/user-attachments/assets/6d7c2546-b989-479a-b962-c01ae9e9d171)
![KakaoTalk_20251122_012426717_07](https://github.com/user-attachments/assets/f250e37e-08d9-4eea-ba73-4fef14ce81ba)
![KakaoTalk_20251122_012426717_08](https://github.com/user-attachments/assets/a29fbcad-0069-4b20-b802-c5db08667f0a)


## Transformer 구조의 응용

### 1. Encoder만 남기면: BERT, ViT (Encoder-Only)
- 공통 구조
  1. Decoder 전체 삭제
  2. Encoder만 N층 쌓음
  3. Self-Attention에서는 양방향: Mask 없이 모든 토큰이 서로를 볼 수 있다.
  4. 최종 출력 = `(B, T, d_model)`

- BERT: Text용 Encoder-only
  1. 입력 토큰
    - 일반적인 NLP 파이프라인: 문장 -> subword token -> 각 토큰 Embedding + Positional Encoding + **Segment Embedding**
  2. Encoder의 동작
    - Transformer에서 사용한 Encoder Layer 그대로 사용
    - Self-Attention시 causal mask 하지 않고 -> 각 토큰이 왼쪽/오른쪽 모든 토큰을 동시에 본다.
  3. 목적
    - MLM(Masked Language Modeling): 일부 토큰을 `Mask`하고 해당 자리에 어떤 단어가 와야할지
    - NSP(Next Sentence Prediction): 문장 A 다음 문장 B가 실제로 이어지는지 아닌지 이진 분류
  -> " 양방향 문맥을 다 본 상태에서 가려진 토큰을 맟주는 것"
  4. 사용 방식
    - Pretrained BERT를 사용하여 원하는 목적에 맞게 구성

- ViT(Vision Transformer): IMG용 Encoder-only
  -> " Text Token 대신 IMG patch를 Token으로 쓴 BERT"
  1. 입력 토큰
     - 이미지 -> 패치
     - 예) 224X224X3 IMG를 PXP 패치로 잘라서 (224/16 = 14 -> 14X14 = 19개의 패치)
     - 각 패치를 벡터로 펼친 뒤 Linear projection
     - 이 patch embedding이 ViT의 토큰 임베딩
     - `CLS`(클래스 토큰) + Patch Embedding + Positional Encoding
  2. Encoder의 동작
     - BERT와 동일: Encoder Layer Stack + 양방향 Self-Attn.
  3. 목적
     - 초기엔 '단순한 이미지 분류 지도 학습' -> '자기지도 학습'

### 2. Decoder만 남기면: GPT (Decoder-Only)
- "Decoder만 남긴 구조로 Cross-Attn도 같이 삭제" -> (Masked Self-Attn + FFN)들 Stack
- GPT의 입력: text sequence -> subword 토큰화 -> 각 토큰 embedding + positional embedding(learnable)
- Self-Attn은 항상 causal mask 사용 (Train, Inference 둘 다)
- 목적: 순수하게 자기회귀 언어 모델링(Autoregressive Language Modeling)
  - 입력이 `[w1, w2, w3, w4]`면
  - 각 위치에서 `[w2]`일 때 -> `w3` 맞추기
  - "항상 왼쪽에서 오른쪽으로 한 칸씩 예측하는 형태
- 추론 시: 한 토큰씩 생성 
