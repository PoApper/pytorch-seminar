---
title: "Homework 1 (Solution)"
layout: post
use_math: true
tags: ["solution"]
---

<br/>

👉 [Homework 1]({{"/2021/10/31/homework-1.html" | relative_url}})

# P1

[colab notebook](https://colab.research.google.com/drive/1VSUYbfsQqwK8DKh2LNYssywpAF7m2t88?usp=sharing)

HW2의 MLP 부분에 대한 답도 함께 작성되어 있다.

Regression 모델의 경우, $R^2$으로 성능을 평가하는게 일반적이다. MSE 또는 RMSE는 데이터셋에 dependent 할 가능성이 크기 때문에 $R^2$로 성능을 리포트해야 한다. (MSE는 NN 모델의 loss 역할로, 또 학습 경향이 어떤지 파악하는 수단 정도로 생각하면 편할 듯)

$R^2$ 값은 single layer의 경우 0.5x, MLP의 경우 0.6x 정도가 나온다. ($R^2$ 0.6x 정도면 그리 높은 편은 아니다 🙏)

# P2

<div>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png" width="200px">
</div>

Sigmoid를 Activation func.로 쓰는 깊은 신경망은 **Gradient Vanishing**이라는 효과가 발생한다. 이는 역전파 과정의 Gradient가 점점 옅어져 입력 단의 Perceptron은 전혀 갱신이 안 되는 문제로 Sigmoid 함수의 양 끝으로 갈수록 기울기가 0이 되는 것이 원인이다. 이를 해결하기 위해 ReLU가 제시되었으며 ReLU는 $x > 0$에 대해 grad 값이 1이기 때문에 backward 되는 grad를 그대로 흘려보내며, 이는 gradient vanishing 효과를 없애준다. 그외에 sigmoid는 식에 exponential이 있어 계산이 복잡하지만, relu는 $\max$ 뿐이라서 계산 상으로 이득이다. $x < 0$에서 0을 반환하기 때문에 연산량을 줄여 준다 등등의 이유가 있다.

# P3

1\. Convolution layer는 2차원 커널을 sliding window 방식으로 훑기 때문에 한 픽셀의 주변 값에 대한 정보를 함께 활용할 수 있다. 이것은 픽셀 하나하나 씩 일일이 보는 FC layer에 비해 2D 배열인 이미지가 갖는 공간 정보를 더 잘 활용하게 한다.

또, 만약 이미지를 FC layer로 학습시키려 한다면 $H \times W$ 만큼의 weight가 필요할 것이다. 이것은 이미지 크기가 커질 수록 필요한 weight가 많아짐을 의미한다. 반면에 Conv. layer는 $N \times M$의 고정된 kernel size 만큼의 weight만 있으면 되기 때문에 계산상으로도 FC layer에 비해 가볍다. 같은 맥락으로 FC layer는 이미지 크기를 $H \times W$로 고정된 이미지가 입력받을 수 있지만, Conv layer를 쓰면 입력받는 이미지에 제약이 없다.

2\. **padding**을 쓰는 이유는 (1) 이미지 가장자리에 대한 정보 손실을 줄이기 위해 (2) Conv layer 적용하면 입력 이미지에 비해 출력 이미지(feature map)의 크기가 약간 작아지게 되는데 padding을 하면 이 줄어드는 크기를 줄일 수 있다.

**pooling**은 sub sampling으로 이미지 크기를 줄여준다. 그래서 pooling을 거치면 연산량이 줄어들어 이득이다. 또, feature map의 특징적인 부분을 뽑거나(max pool), 특정 부분이 튀는 것(noise) 같은 것들의 효과를 억제하는(avg pool) 등을 얻어 모델이 너무 세밀한 것까지 학습해 overfitting 되는 것을 막아준다.

첨언) pooling에서 overfitting에 대한 내용은 조금 empirical한, 경험적인 느낌이 있다. 원래 딥러닝 쪽이 완벽히 이론적으로 설명하는게 아니라 경험적으로 그렇다 하는 부분들이 좀 있다 🙏

