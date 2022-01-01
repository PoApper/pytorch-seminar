---
title: "Seminar 7: RNN and LSTM"
layout: post
use_math: true
tags: ["seminar"]
---

<br/>

이번 포스트에서는 sequential data를 처리하는 모델인 RNN과 LSTM에 대해 살펴본다. 이론적인 부분은 본인이 예전에 작성했던 [RNN & LSTM](https://bluehorn07.github.io/computer_science/2021/07/05/RNN-and-LSTM.html) 포스트를 바탕으로 진행한다. 본 포스트에서는 이론 후 실습하는 PyTorch 코드를 주로 다룬다.

또, 이번 주제는 [Stanford CS231N: Recurrent Neuaral Network](http://cs231n.stanford.edu/2017/syllabus.html#:~:text=Recurrent%20Neural%20Networks)의 수업을 함께 시청하는 것을 추천한다.

<hr/>

## RNN

### 이론

[RNN & LSTM](https://bluehorn07.github.io/computer_science/2021/07/05/RNN-and-LSTM.html) 포스트의 [RNN](https://bluehorn07.github.io/computer_science/2021/07/05/RNN-and-LSTM.html#rnn-recurrent-neural-network) 부분을 참고한다.

(간단 요약)

- RNN은 hidden state $h_t$를 저장해서 이전 입력에 대한 상태를 저장하고 그것을 활용한다.
- RNN 학습은 하나의 시퀀스(sequence) 입력 단위로 이뤄진다. 배치(batch)와는 개념이 조금 다르다.

### 실습: CharRNN

![RNN structure](https://i.stack.imgur.com/b4sus.jpg)

**many-to-many** 구조의 RNN 모델을 사용하면 텍스트를 생성하는 모델을 구현할 수 있다. 이번 세미나에서는 텍스트를 생성하기 위한 학습과 출력의 단위가 문자(character) 단위로 진행해보겠다.

<br/>

먼저 학습에 사용할 텍스트를 정해보자. 아래는 괴테의 문구 중 하나다.

```py
# famous sentence of Johann Wolfgan von Goethe
text = 'Knowing is not enough; we must apply. ' \
       'Willing is not enough; we must do'
```

<br/>

다음은 학습 텍스트에 있는 문자들을 인코딩 해야 한다. `dict`을 이용해 아래와 같이 문자 인코딩을 할 수 있다.

```py
# create character set and disctionary
char_set = list(set(text))
char_dict = { c : i for i, c in enumerate(char_set)}
dict_size = len(char_dict)
char_dict
...
{' ': 14,
 '.': 3,
 ';': 19,
 'K': 6,
 'W': 1,
 ...
}
```

이제 이 `char_dict` 인코딩을 이용해 학습할 텍스트를 인코딩 하자. 이때, RNN은 일종의 시퀀스(sequence) 단위로 학습을 진행하는데, `sequence_length=10`으로 설정하고 실제 RNN 학습에 사용할 데이터셋을 준비하자!

```py
sequence_length = 10

x_data = []
y_data = []

for i in range(0, len(text) - sequence_length):
  # slice input text into fixed length samples
  x_str = text[i : i + sequence_length]
  y_str = text[i + 1 : i + sequence_length + 1]
  if i % 10 == 0:
    print(i, x_str, '->', y_str)

  # dictionary encoding
  x_data.append([char_dict[c] for c in x_str])
  y_data.append([char_dict[c] for c in y_str])

print(x_data[0])
print(y_data[0])
```

그런데 지금 상태 그대로는 RNN 학습의 입력으로 넣을 순 없고, one-hot encoding 방식으로 변환해 RNN 모델 입력에 넣어줘야 한다. 그래서 $N\times N$ 행렬을 만드는 `np.exe(N)` 함수를 활용해 아래와 같이 one-hot encoding 해준다.

```py
# one-hot encoding
x_one_hot = np.array([np.eye(dict_size)[x] for x in x_data])

# create tensor dataset
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

print("train data size: {}".format(X.shape)) # (len(x), seq_len, dict_size)
print("label size: {}".format(Y.shape))
```

<br/>

이제 데이터셋이 준비되었으니 charRNN 모델을 구현해보자. 이 모델은 문자 하나를 입력 받고, 다음 문자를 추론하는 모델이다.

```py
class CharRNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, layers):
    super(CharRNN, self).__init__()
    self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def forward(self, x):
    x, _status = self.rnn(x)
    x = self.fc(x)
    return x
```

위의 모델에서는 `nn.RNN()`과 함께 `nn.Linear()`를 사용하고 있다. 물론 `nn.RNN()`만 사용해도 모델을 만들 순 있지만, 직접 해보면 퍼포먼스가 좋지 않아서 `nn.Linear()`를 추가했다.

`forward()` 블록의 `nn.RNN()` 레이어의 출력을 보자. `x, _status`로 나누어져 출력되는데, 각각 RNN의 출력 $y_t$와 RNN의 hidden layer $h_t$를 의미한다.

- `hidden_dim`: $h_t$의 차원
- `num_layers`: RNN 내부의 hidden layer 갯수

<br/>

자! 이제 하이퍼 파라미터를 정하고, 학습을 시켜보자!

```py
# hyper-parameter
learning_rate = 0.1
num_layers = 2

model = CharRNN(dict_size, dict_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# debug model output
outputs = model(X)
print(dict_size)
print(outputs.shape) # (batch, timesteps, dict_size)
```

```py
def textimize_result(results):
  predict_str = ""

  for j, result in enumerate(results):
    if j == 0: # At first time, bring all results
      predict_str += ''.join([char_set[t] for t in result])
    else: # append only last character
      predict_str += char_set[result[-1]]
  print(predict_str)

epoch = 51

for i in range(epoch):
  optimizer.zero_grad()
  
  outputs = model(X) # (batch, timesteps, dict_size)
  loss = criterion(outputs.reshape(-1, dict_size), Y.view(-1))
  
  loss.backward()
  optimizer.step()

  results = outputs.argmax(dim = 2) # (len(text), timesteps)

  if i % 10 == 0:
    print("------ %d ------" % i)
    textimize_result(results)
```

`textimize_result()` 함수는 추론 결과를 concat 하여 text 형태로 보여주는 함수다. 내부적인 구현은 중요치 않다.

```text
------ 0 ------
mssss;sssssss;;s;s;;;;sss;sss;sms;;;;sss;;sssssssss;;s;s;;;;sss;sss;ss
------ 10 ------
ng  n wus  g ;e mugh; we mus  ul lsmu;on lug ;s  g ;e mugh; we mus  uo
------ 20 ------
ngtnng is not enough; we must apply. Willing is not enough; we must ao
------ 30 ------
nowing is not enough; we must apply. Willing is not enough; we must ao
------ 40 ------
nowing is not enough; we must apply. Willing is not enough; we must ao
------ 50 ------
nowing is not enough; we must apply. Willing is not enough; we must ao
```

와우! 초반에는 출력이 정말 이상했는데, 학습을 거듭할 수록 원본 텍스트를 거의 비슷하게 추론하게 된다!

## LSTM

### RNN의 한계

hidden state $h_t$로 이전 상태를 저장하는 RNN 모델이다. 그러나 $h_t$는 가까운 이전 상태만을 기억할 뿐. 더 이전의 문맥적인 요소는 고려하지 못한다는 단점이 있다. 그래서 입력 데이터가 길어진다면 RNN의 효율이 떨어진다. 이를 극복하기 위해 새로운 sequential model인 LSTM이 제시되었다.

### 이론

[RNN & LSTM](https://bluehorn07.github.io/computer_science/2021/07/05/RNN-and-LSTM.html) 포스트의 [LSTM](https://bluehorn07.github.io/computer_science/2021/07/05/RNN-and-LSTM.html#lstm-long-short-term-memory-model) 부분을 참고한다.

(간단 요약)

- LSTM은 장기 기억을 담당하는 cell state $c_t$와 단기 기억을 담당하는 hidden state $h_t$로 구성된다.
- LSTM은 4가지 gate를 통해 입력된 정보를 정제하고, 장기 기억의 일부를 망각하거나, 장기 기억의 일부를 단기 기억으로 전환한다.

### 실습

LSTM 모델은 RNN 모델보다 더 긴 텍스트를 처리하는데 적합하다.

이번에는 좀더 긴 텍스트를 입력으로 사용해보자. 미국의 국가를 입력 데이터로 사용해보자.

```py
# US National Anthem
text = f'''O say can you see, by the dawn’s early light,
What so proudly we hail’d at the twilight’s last gleaming,
Whose broad stripes and bright stars through the perilous fight
O’er the ramparts we watch’d were so gallantly streaming?
And the rocket’s red glare, the bombs bursting in air,
Gave proof through the night that our flag was still there,
O say does that star-spangled banner yet wave
O’er the land of the free and the home of the brave?'''
```

모델은 기존의 `CharRNN`에서 `nn.LSTM()`으로 바꿔주기만 하면 된다.

```py
class CharLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, layers):
    super(CharLSTM, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def forward(self, x):
    x, _status = self.lstm(x)
    x = self.fc(x)
    return x

model = CharLSTM(dict_size, dict_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
```

학습 과정 역시 `CharRNN` 때와 동일하며, 출력 결과는 아래와 같다.

```text
------ 0 ------
 hhh hhhhhh  hhhhhhhhhh hhhhh    hhhh hhh       h h  hhhhhhh h  hhhhh hhh     h           hh      h           hhhhhhhh   hh hhhhhhhhh        h  h h  h  hhhhhhh   hhh        h  hhhhhhhh     h       hhhhhhh h    hh                 hhhhhhhhhhh    hhhhhhhhhhh  hhhhhhhhhhhh      hhhhh     hhhhh  h  h h  h  hhhh           hhhhh h   h     h  hhhh  hhh hhhh h         h hhh   hhhhhhhhhhh           h  hhhhhhhhhh  hhhhhhhhhhhh hhhhhhhhhh  hhhhhhh
------ 20 ------
 a e aaeea eta    aa a e a etta a e a eae e    ea a aaa at a ee aeteaa ae a e a eeae e a aae  e aaeaet    e   aa ae a  sea  aee ea e e a  e aa e at ea e aa  eaat a e e s a  a e a etae   ae aet aea ae  aa aa eaaee a e  saeaet eaee a e a aae  e a a a ae a a e aaaaa aat a et aetaee aaaee at aa e e at ea e aee e a et eat e ae aet a  eaat e  a   a e a a  a et e  e aaaee aa aaeet  a a eete a a  a e aaee ea e e a aa aee a e ae a aa e e aa ee 
------ 40 ------
 ttt tandt udoae  ti the tolr s tedoa tanht  What sa taaud o te tatglt tn she thaglght s tag othaetindh hhaue tiaun stoonat tnd tiaght saogh theaudh the tarogiud toght o e  the taniatah te tat   t te   ta ehtiagd o taooatindh hnd the tauhet s tar taana  the tiusi tirothgdhsndain  hate lrauu theaudh the tight shet surotiaghsat taoglathe    o ttt to e thet saoghaaagghaa tinder t e tete s e  the tand su the toae tnd the ta e tu the tiage 
------ 60 ------
 sti san aou ste, bi the rown’s earoy waght

What st prougly we sail’d at the raalight’s last glaaming 
Whase bross stiipes and biight stir- through the berelaug blght
O’er the bamparts we eatch’d we e bo gallantlo wtreaming,
Wnd the bauket’s lad glare, the bimss bursting sn air,
Oave prouu theough the bight
thet sur yrag sas stil’ the e, O sti soes that stir-spanglad winner eot wave
O’er the band au the bree snd the bame su the brote 
------ 80 ------
’say dan you soe, by the bown’s earey light,
Ohat so proudly we hail’d at the railight’s last glaaming,
Whose broad stripes and bright stars through the berolous fight
O’er the ramparts we watch’d were so gallantly streaming,
Wnd the racket’s red glare, the bombs bursting hn air,
Wave proof through the bight that sur flag was still there,
O say does that starsspangled wanner yet wave
O’er the rand of the bree and the rome of the braveh
------ 100 ------
’say dan you see, by the rawn’s early light,
What so proudly we hail’d at the bailight’s last glaaming,
Whose broad stripes and bright stars through the berolous fight
O’er the lamparts we watch’d were so gallantly streaming,
Wnd the rocket’s red glare, the bombs bursting in air,
Wave proof through the bight that our flag was still there,
O say does that star-spangled banner yet wave
O’er the land of the bree and the home of the brave?
```

<hr/>

## 맺음말

이번 세미나에서 다룬 RNN과 LSTM은 딥러닝의 자연어처리 모델에서 가장 기본이 되는 모델이다. 또, 이번 세미나에서는 문자(character) 수준의 모델을 만들었지만, 단어(word), 문장(sentence) 단위의 sequential model을 만들기 위해선 더 많은 테크닉을 살펴봐야 한다.

다음 세미나에서는 Word Embedding과 Word2Vec 모델을 살펴보자!

<hr/>

## References

- [PyTorch로 시작하는 딥러닝 입문](https://wikidocs.net/64739)


