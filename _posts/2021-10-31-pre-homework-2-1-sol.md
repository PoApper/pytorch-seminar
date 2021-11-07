---
title: "Pre-homework 2-1 (Solution)"
layout: post
use_math: true
tags: ["solution"]
---

<br/>

# P1. Linear Regression

[Least Square Method](https://bluehorn07.github.io/mathematics/2021/06/06/introduction-to-linear-regression.html#least-square-method) 포스트의 내용 참고

# P2. GD & Back-prop

1\. False: linear regression에서는 GD가 늘 global optimum을 찾는다.

> Gradient Descent is an algorithm which is designed to find the optimal points, but these optimal points are not necessarily global.

![](https://wingshore.files.wordpress.com/2014/11/images-12.jpg)

> Informally a Convex Function is considered a bow shaped function and hence, this function does not have any local optima. Therefore, it has only one Global Optimum.Thus, when we use Gradient descent to the Linear Regression model, then we will notice that it will always converge to the Global Minimum. Since, there is no local optima for this!

설명1) [Convex Cost function and Convex Problem](https://stackoverflow.com/a/41594623/11983825)

설명2) [Hessian을 이용해 Convex를 판단](https://math.stackexchange.com/questions/2774106/why-is-the-least-square-cost-function-for-linear-regression-convex). [$X^TX$ is always positive semin-definite](https://statisticaloddsandends.wordpress.com/2018/01/31/xtx-is-always-positive-semidefinite/) 

일반적으로 non-linear hidden layer가 있는 NN의 경우 multiple local minima를 갖습니다.

<br/>

2\. Batch GD는 데이터셋 전체에 대해 GD를 구한 후 parameter를 갱신, stochastic GD는 데이터 하나에 대해 GD를 구한 후 parameter를 갱신. Batch GD는 전체 데이터셋을 모두 봐야 한다 만약 train-set이 어마어마하게 크다면 train-set만 보는데 엄청난 시간이 소요된다. 그러나 stochastic GD는 train-set의 일부를 보기 때문에 iteration을 도는 시간이 빠르고 이에 따라 GD보다 빠르게 수렴할 가능성이 있다.

<br/>

3\. [Back-propagation](https://bluehorn07.github.io/computer_science/2021/05/23/back-propagation.html) 포스트의 예시 문제 참고


