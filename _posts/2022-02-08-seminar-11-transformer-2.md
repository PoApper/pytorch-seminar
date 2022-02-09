---
title: "Seminar 11: Transformer (구현)"
layout: post
use_math: true
tags: ["seminar"]
---

<br/>

이번 포스트에서는 PyTroch의 `nn.Transformer` 모듈을 활용해 Transformer seq2seq 번역 모델을 구현해보겠습니다. 실습 코드는 PyTorch Tutorials의 ["NN.TRANSFORMER와 TORCHTEXT로 언어 번역하기"](https://tutorials.pytorch.kr/beginner/translation_transformer.html) 문서를 참고했음을 밝힙니다.

<hr/>

# `nn.Transformer`로 독일어-영어 번역

Pytorch의 `torchtext`에서는 [Multi30k](https://github.com/multi30k/dataset)라는 데이터셋을 제공합니다. Multi30k는 영어, 독일어 문장 쌍을 제공합니다. 이번 실습에서는 요 데이터셋을 활용해 독일어-영어 번역 모델을 구현해보겠습니다.

## 사전 준비

시작하기 전에 [spacy](https://wikidocs.net/64517)라는 tokenizer를 설치하겠습니다. 

```bash
!pip install -U spacy
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
```

## 데이터셋

`torchtext`의 [Multi30k](https://pytorch.org/text/stable/datasets.html#multi30k)를 사용하겠습니다. torchtext는 데이터셋을 iterator를 통해 제공합니다.

```py
# 사용할 데이터셋 확인
## 학습용 데이터 반복자
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

train_iter = Multi30k(root='./data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

for idx, data_sample in enumerate(train_iter):
  if idx > 10: break
  print(idx, data_sample)
------------------------------------
1 ('Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.\n', 'Several men in hard hats are operating a giant pulley system.\n')
2 ('Ein kleines Mädchen klettert in ein Spielhaus aus Holz.\n', 'A little girl climbing into a wooden playhouse.\n')
3 ('Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.\n', 'A man in a blue shirt is standing on a ladder cleaning a window.\n')
```

## 전처리

다음으로는 언어별 tokenizer를 정의해 `token_transform` 딕셔너리에 저장하겠습니다. 이후에 vocab set을 만들 때와 `Dataloader` 단에 `collate_fn`라는 인자에 사용할 예정입니다. (*collate*는 '수집하다/분석하다'라는 뜻의 단어 입니다만, 대충 `DataLoader`에서 쓰는 transform의 용도로 사용된다고 받아들이시면 될 것 같습니다.)

```py
# tokenizer
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
```

<br/>

토큰화에 사용할 몇가지 특수 토큰을 정의합니다. 이후에 vocab set을 만들 때 사용합니다.

```py
# 특수 토큰
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

UNK_TKN_IDX = 0
PAD_TKN_IDX = 1
SOS_TKN_IDX = 2
EOS_TKN_IDX = 3
```

<br/>

다음으로는 `torchtext`의 `build_vocab_from_iterator()` 함수를 사용해 vocab set을 생성하겠습니다.

```py
# vocabulary set(어휘집) 생성
vocab_transform = {} # integer encoding을 수행하는 vocab set의 묶음

# (helper function) 데이터셋에서 특정 language의 것만 반환하는 generator function.
# `yield` 키워드에 주목하자.
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
  language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
  tokenizer = token_transform[language]
  
  for data_sample in data_iter:
    sentence = data_sample[language_index[language]]
    yield tokenizer(sentence)

# vocab set: SRC LANGUAGE
train_iter = Multi30k(root='./data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
vocab_transform[SRC_LANGUAGE] \
  = build_vocab_from_iterator(
      iterator = yield_tokens(train_iter, SRC_LANGUAGE),
      min_freq = 1,
      specials = special_symbols,
      special_first = True # `specials`의 토큰이 가장 앞의 index를 가지도록 설정
    )
vocab_transform[SRC_LANGUAGE].set_default_index(UNK_TKN_IDX)
print(vocab_transform[SRC_LANGUAGE](token_transform[SRC_LANGUAGE]("Eine Gruppe von Menschen steht vor einem Iglu .")))
-----------------------------
[15, 39, 25, 55, 31, 29, 7, 6133, 5]
```

이때 생성되는 vocab set은 `toechtext.vocab.Vocab` 객체인데, `get_itos()`, `get_stoi()` 등의 함수로 vocab set의 기능을 제공합니다. [pytorch docs](https://pytorch.org/text/stable/vocab.html#vocab) 앞에서 만든 `token_transform`와 마찬가지로 요 녀석도 `DataLoader`의 `collate_fn` 인자로 사용됩니다.

## 모델링

Transformer 번역 모델을 만들기 위해 크게 3가지 작업에 대한 코드를 작성해야 합니다.

1. Embedding: token embedding & positional encoding
2. Transformer
3. Ouput Layer

### Embedding

#### Token Embedding

`nn.Embedding()` 레이어를 통해 토큰(단어)를 임베딩하는 모듈입니다. 이전의 Word Embedding의 것을 모듈화한 것에 불과합니다.

```py
class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size: int, emb_size: int):
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.emb_size = emb_size
  
  def forward(self, tokens: Tensor):
    out = self.embedding(tokens.long())
    out *= math.sqrt(self.emb_size) # scaling
    return out
```

#### Positional Encoding

Transformer 논문의 positional encoding 수식을 모듈로 구현합니다.

$$
PE(pos) = 
\begin{cases}
  \sin(\omega_k \cdot pos) & \text{if} \; i = 2k \\
  \cos(\omega_k \cdot pos) & \text{if} \; i = 2k+1 
\end{cases} 
\quad \left( \omega_k = \frac{1}{1000^{k/d}} \right)
$$

frequency인 $\omega_k = \frac{1}{1000^{k/d}}$를 구할 때 underflow가 나는 걸 방지하기 위해 log로 변환하는 트릭을 사용합니다.

Note: $\omega_k = \frac{1}{1000^{k/d}} = \exp \left( \log \frac{1}{1000^{k/d}} \right) = \exp \left( - \log (1000^{k/d}) \right) = \exp \left( - \log (1000) \times {k/d} \right)$

```py
class PositionalEncoding(nn.Module):
  def __init__(self, emb_size:int, maxlen: int = 5000):
    super(PositionalEncoding, self).__init__()
    
    pos = torch.arange(0, maxlen).reshape(maxlen, 1)

    # $1000^{k/d}$를 실제로 계산하려고 하면 underflow가 발생할 수 있으니 log trick을 사용
    frequency = torch.exp(- math.log(1000) * (torch.arange(0, emb_size, 2) / emb_size))

    # sinusoidla encoding by sin & cos
    pos_embedding = torch.zeros((maxlen, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * frequency) # 짝수 인덱스
    pos_embedding[:, 1::2] = torch.cos(pos * frequency) # 홀수 인덱스
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.register_buffer('pos_embedding', pos_embedding) # ref. https://powerofsummary.tistory.com/158
  ...
```

코드 중에 `self.register_buffer('pos_embedding', pos_embedding)`라는 부분은 `optimizer`로 업데이트하지 않는 grad option이 꺼진 layer를 모듈에 등록합니다. 이는 `transformer.parameters()`과 같이 모델의 전체 파라미터를 순회할 때 `pos_embedding` 레이어 역시 함께 순회하기 위함입니다. 이후에 등장하는 코드에서 한번 더 `register_buffer()` 함수의 기능을 리마인드 하겠습니다.

<br/>

```py
class PositionalEncoding(nn.Module):
  ...
  def forward(self, token_embedding: Tensor):
    token_length = token_embedding.size(0)
    return token_embedding + self.pos_embedding[:token_length, :]
```

`token_embedding + self.pos_embedding[:token_length, :]` 부분을 보면 논문에서 처럼 embedding vector와 positional encoding 값을 더해 position information을 전달하는 걸 볼 수 있습니다.

#### Seq2SeqTransformer

`nn.Transformer()` 모듈을 사용해 Seq2Seq를 수행하는 모델을 구현합니다. 

```py
class Seq2SeqTransformer(nn.Module):
  def __init__(self, 
               num_encoder_layers: int, num_decoder_layers:int, 
               emb_size: int, nhead:int,
               src_vocab_size:int, tgt_vocab_size:int,  
               dim_feedforward:int = 512):
    super(Seq2SeqTransformer, self).__init__()
    
    self.src_tkn_emb = TokenEmbedding(src_vocab_size, emb_size)
    self.tgt_tkn_emb = TokenEmbedding(tgt_vocab_size, emb_size)
    self.positional_encoding = PositionalEncoding(emb_size)

    self.transformer = nn.Transformer(
        d_model = emb_size,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_decoder_layers,
        dim_feedforward = dim_feedforward)

    self.output_layer = nn.Linear(emb_size, tgt_vocab_size)
```

`TokenEmbedding`, `PositionalEncoding`, `nn.Transformer` 레이어를 정의합니다.

<br/>

```py
class Seq2SeqTransformer(nn.Module):
  ...
  def forward(self, 
              src: Tensor, tgt: Tensor,
              src_mask: Tensor, tgt_mask: Tensor,
              src_pad_mask: Tensor, tgt_pad_mask: Tensor):
    # embedding
    src_emb = self.positional_encoding(self.src_tkn_emb(src))
    tgt_emb = self.positional_encoding(self.tgt_tkn_emb(tgt))

    # transformer
    outs = self.transformer(
        src_emb, tgt_emb, 
        src_mask, tgt_mask, None, 
        src_pad_mask, tgt_pad_mask, src_pad_mask
      )

    # output layer
    out = self.output_layer(outs)
    return out
```

`transformer` 레이어를 사용하는 부분에서 `... src_pad_mask, tgt_pad_mask, src_pad_mask`로 `src_pad_mask`를 한번 더 사용했다. `nn.Transformer()` [문서](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#transformer)를 보면 `memory_key_padding_mask` 인자에 해당하는 부분을 Encoder-Decoder Attention을 할 때 Key로 사용하는 Encoder의 vector에 쓰는 pad mask에 해당하는 인자다. 그래서 `src_pad_mask`를 그대로 사용한다.

<br/>

```py
class Seq2SeqTransformer(nn.Module):
  ...
  def encode(self, src: Tensor, src_mask: Tensor):
    src_emb = self.positional_encoding(self.src_tkn_emb(src))
    return self.transformer.encoder(src_emb, src_mask)

  def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
    tgt_emb = self.positional_encoding(self.tgt_tkn_emb(tgt))
    return self.transformer.decoder(tgt_emb, memory, tgt_mask)
```

따로 구현한 2개의 함수 `encode()`와 `decode()`는 각각 encoding과 decoding 과정을 별도의 함수로 분리한 것으로 모델 학습 후에 demo 과정에서 사용할 때 사용하게 된다.

## 학습

### 하이퍼 파라미터

```py
torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

BATCH_SIZE = 128
```

### 모델 선언

```py
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
  if p.dim() > 1:
      nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TKN_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
```

[앞에서](#positional-encoding) `PositionalEncoding` 모듈을 구현할 때 `register_buffer()` 함수를 사용했었다. 만약 이 함수가 없다면 `PositionalEncoding` 모듈에 `positional_encoding` 레이어가 파라미터로 포함되지 않아서 `transformer.to(DEVICE)`할 때 `positional_encoding` 레이어는 그대로 `cpu` 디바이스에 남게 되는 문제가 발생한다.

### 문자열 → 배치 텐서

```py
# raw한 문자열 배치를 배치 텐서로 조합(collate)하는 함수
## 기본틀
def collate_fn(raw_batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in raw_batch:
    ...
  return src_batch, tgt_batch
```

처음의 [데이터셋](#데이터셋)을 디버그할 때 봤듯이 데이터 반복자(iterator)는 raw한 문자열의 쌍을 생성합니다. 이 문자열 쌍들을 정의한 Transformer에서 쓸 수 있도록 텐서 묶음(batch tensor)으로 변환해야 합니다. `collate_fn()` 함수는 이런 기능을 수행합니다!

<br/>

```py
def collate_fn(raw_batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in raw_batch:
    src_sample = src_sample.rstrip("\n")
    src_sample = token_transform[SRC_LANGUAGE](src_sample) # tokenize
    src_batch.append(src_sample)
  return src_batch, tgt_batch

train_iter = Multi30k(root='./data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

for idx, sample in enumerate(train_dataloader):
  if idx > 2: break
  print(idx, sample)
-----------------------------------
0 ([['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der',  ....
1 ([['Ein', 'Typ', ',', 'der', 'blau', 'trägt', ',', 'in', 'einem', 'Loch',  ...
2 ([['Zwei', 'Personen', 'arbeiten', 'daran', ',', 'den', 'Schnee', 'von', ...
```

처음의 [전처리](#전처리) 항목에서 정의한 `token_transform`을 사용합니다. `collate_fn`에서 tokenizer를 적용하기 때문에 `train_dataloader`에서 받는 배치 데이터가 tokenized 되어 있는 것을 볼 수 있습니다.

<br/>

```py
def collate_fn(raw_batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in raw_batch:
    src_sample = src_sample.rstrip("\n")
    src_sample = token_transform[SRC_LANGUAGE](src_sample) # tokenize
    src_sample = vocab_transform[SRC_LANGUAGE](src_sample) # integer encoding
    src_batch.append(src_sample)
  return src_batch, tgt_batch

...
-----------------------------------
0 ([[22, 86, 258, 32, 88, 23, 95, 8, 17, 113, 7911, 3210, 5], ...
1 ([[6, 457, 9, 17, 310, 63, 9, 8, 7, 777, 5], [130, 68, 329, ...
2 ([[22, 43, 188, 2222, 9, 35, 126, 25, 7, 462, 30, 2228, 5], ...
```

마찬가지로 `vocab_transform`를 사용해 integer encoding 된 배치 데이터를 받도록 합니다.

<br/>

```py
def collate_fn(raw_batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in raw_batch:
    src_sample = src_sample.rstrip("\n")
    src_sample = token_transform[SRC_LANGUAGE](src_sample) # tokenize
    src_sample = vocab_transform[SRC_LANGUAGE](src_sample) # integer encoding
    src_sample = torch.cat((torch.tensor([SOS_TKN_IDX]),
                      torch.tensor(src_sample),
                      torch.tensor([EOS_TKN_IDX])))
    src_batch.append(src_sample)
  return src_batch, tgt_batch

...
------------------------------------
0 torch.Size([15]) tensor([   2,   22,   86,  ...    5,    3])
```

문장의 양 끝에 `<sos>` 토큰(인코딩 2번)과 `<eos>` 토큰(인코딩 3번)을 붙여줍니다.

<br/>

```py
from torch.nn.utils.rnn import pad_sequence

def collate_fn(raw_batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in raw_batch:
    src_sample = src_sample.rstrip("\n")
    src_sample = token_transform[SRC_LANGUAGE](src_sample) # tokenize
    src_sample = vocab_transform[SRC_LANGUAGE](src_sample) # integer encoding
    src_sample = torch.cat((torch.tensor([SOS_TKN_IDX]),
                      torch.tensor(src_sample),
                      torch.tensor([EOS_TKN_IDX])))
    src_batch.append(src_sample)
  
  src_batch = pad_sequence(src_batch, padding_value=PAD_TKN_IDX)

  return src_batch, tgt_batch

...
------------------------------------
0 torch.Size([27, 128]) (tensor([[ 2,  2,  2,  ...,  2,  2,  2],
        [22, 85,  6,  ..., 22, 15, 15],
        [86, 32, 70,  ..., 47, 39, 18],
        ...,
        [ 1,  1,  1,  ...,  1,  1,  1],
        [ 1,  1,  1,  ...,  1,  1,  1],
        [ 1,  1,  1,  ...,  1,  1,  1]]), [])
```

`pad_sequence()` 함수를 사용하면 배치에 `<pad>` 토큰(인코딩 1번)을 넣을 수 있습니다. 결과를 보면 인코딩된 문장의 말미가 모두 `<pad>` 토큰으로 설정 되어 있는 걸 볼 수 있습니다.

<br/>

동일 작업을 `TGT_LANGUAGE`에도 적용한 최종적인 `collate_fn()`은 아래와 같습니다.

```py
from torch.nn.utils.rnn import pad_sequence

def collate_fn(raw_batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in raw_batch:
    src_sample = src_sample.rstrip("\n")
    src_sample = token_transform[SRC_LANGUAGE](src_sample) # tokenize
    src_sample = vocab_transform[SRC_LANGUAGE](src_sample) # integer encoding
    src_sample = torch.cat((torch.tensor([SOS_TKN_IDX]),
                      torch.tensor(src_sample),
                      torch.tensor([EOS_TKN_IDX])))
    src_batch.append(src_sample)

    tgt_sample = tgt_sample.rstrip("\n")
    tgt_sample = token_transform[TGT_LANGUAGE](tgt_sample) # tokenize
    tgt_sample = vocab_transform[TGT_LANGUAGE](tgt_sample) # integer encoding
    tgt_sample = torch.cat((torch.tensor([SOS_TKN_IDX]),
                      torch.tensor(tgt_sample),
                      torch.tensor([EOS_TKN_IDX])))
    tgt_batch.append(tgt_sample)
  
  src_batch = pad_sequence(src_batch, padding_value=PAD_TKN_IDX)
  tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_TKN_IDX)

  return src_batch, tgt_batch
```

### Mask 생성

Transformer 논문에 제시된 pad mask와 subsequent mask를 구현해봅시다.

#### pad mask

```py
# pad mask부터 구현
def create_mask(src: Tensor, tgt: Tensor):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  src_pad_mask = (src == PAD_TKN_IDX).transpose(0, 1) # transpose(0, 1): make batch dim first
  tgt_pad_mask = (tgt == PAD_TKN_IDX).transpose(0, 1)

  return src_pad_mask, tgt_pad_mask

train_iter = Multi30k(root='./data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

for idx, sample in enumerate(train_dataloader):
  if idx >= 1: break
  print(idx, sample[0].shape) # (max_seq_len, batch_size)
  src_pad_mask, tgt_pad_mask = create_mask(sample[0], sample[1])
  print(src_pad_mask.shape, tgt_pad_mask.shape) # (batch_size, max_seq_len)
---------------------------
0 torch.Size([27, 128])
torch.Size([128, 27]) torch.Size([128, 24])
```

#### subsequent mask

```py
def generate_square_subsequent_mask(size: int):
  mask = torch.triu(torch.ones((size, size), device=DEVICE) == 1) # upper triangular
  mask = mask.transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf'))
  mask = mask.float().masked_fill(mask == 1, float(0.0))
  return mask

# subsequent mask도 구현
def create_mask(src: Tensor, tgt: Tensor):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  src_pad_mask = (src == PAD_TKN_IDX).transpose(0, 1) # transpose(0, 1): make batch dim first
  tgt_pad_mask = (tgt == PAD_TKN_IDX).transpose(0, 1) # (batch_size, max_seq_len)

  src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool) # no need!
  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

  return src_mask, tgt_mask, src_pad_mask, tgt_pad_mask

for idx, sample in enumerate(train_dataloader):
  if idx >= 1: break
  print(idx, sample[0].shape) # (max_seq_len, batch_size)
  src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(sample[0], sample[1])
  print(src_mask.shape, tgt_mask.shape) # (max_seq_len, max_seq_len)
  print(src_pad_mask.shape, tgt_pad_mask.shape) # (batch_size, max_seq_len)
-------------------
0 torch.Size([33, 128])
torch.Size([33, 33]) torch.Size([36, 36])
torch.Size([128, 33]) torch.Size([128, 36])
```

`src` 텐서의 경우는 `src_mask`가 딱히 필요 없기 때문에 `torch.zeros()`로 별도로 필터링 하지 않는다. `tgt` 텐서의 경우, `torch.triu()`를 통해 미래의 토큰을 볼 수 없도록 가려둔다.

### Train Epoch

먼저 큰 틀은 아래와 같다. 아래의 코드에서 조금씩 발전시켜 보겠다.

```py
# data load
def train_epoch(model, optimizer):
  model.train()
  total_loss = 0
  
  train_iter = Multi30k(root='./data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  for src, tgt in train_dataloader:
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)
    break

  return total_loss

NUM_EPOCHS = 18
for epoch in range(1, NUM_EPOCHS+1):
  train_loss = train_epoch(transformer, optimizer)
  print(epoch, train_loss)
  break
```

<br/>

```py
def train_epoch(model, optimizer):
  ...
  for src, tgt in train_dataloader:
    ...
    tgt_input = tgt[:-1, :] # remove last <pad> token
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
    ...    
```

먼저 mask들을 생성한다.

<br/>

```py
def train_epoch(model, optimizer):
  ...
  for src, tgt in train_dataloader:
    ...
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
    ...
```

Transformer `model`에 넣어 logit(로짓)을 구한다. logit은 쉽게 말해 확률(probability)이랑 비슷하지만 다른 녀석인데, 일단 대충은 logit이 클수록 해당 토큰의 확률이 높다고 이해하자! 자세한 내용은 이곳의 [포스트](https://velog.io/@guide333/logit-%ED%99%95%EB%A5%A0-sigmoid-softmax)을 참고

<br/>

```py
def train_epoch(model, optimizer):
  ...
  for src, tgt in train_dataloader:
    ...
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
    
    optimizer.zero_grad()

    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    total_loss += loss.item()
    ...
```

마지막으로 `criterion`와 `optimizer`으로 모델을 업데이트 한다.

<br/>

최종적인 형태는 아래와 같다.

```py
# data load
def train_epoch(model, optimizer):
  model.train()
  total_loss = 0
  
  train_iter = Multi30k(root='./data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  for src, tgt in train_dataloader:
    src = src.to(DEVICE) # (max_seq_len, batch_size)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :] # remove <eos> token?
    src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)

    optimizer.zero_grad()

    tgt_out = tgt[1:, :]
    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    total_loss += loss.item()

  return total_loss / len(train_dataloader)


tic = time.time()
epoch_loss = train_epoch(transformer, optimizer)
toc = time.time()
print(f'time: {toc - tic:5.1f} sec | train loss: {epoch_loss:8.4f}')
--------------------------------
time:  36.1 sec | train loss:   5.2571
```

### Evaluate

```py
def evaluate(model):
  model.eval()
  total_loss = 0

  val_iter = Multi30k(root='./data', split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  for src, tgt in val_dataloader:
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]
    src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)

    tgt_out = tgt[1:, :]
    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    total_loss += loss.item()

  return total_loss / len(val_dataloader)


tic = time.time()
val_loss = evaluate(transformer)
toc = time.time()
print(f'time: {toc - tic:5.1f} sec | val loss: {val_loss:8.4f}')
---------------------
time:   0.8 sec | val loss:   3.9802
```

`train_epoch()` 함수에서 모델 갱신 부분만 빼주면 된다.

### Epoch Train

```py
NUM_EPOCHS = 15

for epoch in range(1, NUM_EPOCHS+1):
  tic = time.time()
  train_loss = train_epoch(transformer, optimizer)
  val_loss = evaluate(transformer)
  toc = time.time()
  print(f'| epoch: {epoch:3d} | time: {toc - tic:5.1f} sec | train loss: {train_loss:6.4f} | val loss: {val_loss:6.4f}')
---------------------
| epoch:   1 | time:  36.8 sec | train loss: 3.5973 | val loss: 3.1816
| epoch:   2 | time:  37.3 sec | train loss: 2.9523 | val loss: 2.7729
| epoch:   3 | time:  37.7 sec | train loss: 2.5324 | val loss: 2.5115
| epoch:   4 | time:  38.0 sec | train loss: 2.2249 | val loss: 2.3437
| epoch:   5 | time:  38.3 sec | train loss: 1.9785 | val loss: 2.2367
| epoch:   6 | time:  38.5 sec | train loss: 1.7801 | val loss: 2.1441
| epoch:   7 | time:  38.6 sec | train loss: 1.6109 | val loss: 2.0773
| epoch:   8 | time:  38.7 sec | train loss: 1.4680 | val loss: 2.0306
| epoch:   9 | time:  38.8 sec | train loss: 1.3422 | val loss: 2.0358
| epoch:  10 | time:  38.8 sec | train loss: 1.2319 | val loss: 2.0368
| epoch:  11 | time:  38.8 sec | train loss: 1.1262 | val loss: 2.0542
| epoch:  12 | time:  38.8 sec | train loss: 1.0322 | val loss: 2.0560
| epoch:  13 | time:  38.9 sec | train loss: 0.9517 | val loss: 2.0292
| epoch:  14 | time:  38.9 sec | train loss: 0.8785 | val loss: 2.0077
| epoch:  15 | time:  39.5 sec | train loss: 0.8028 | val loss: 2.0435
```

## 성능 확인

```py
# 순차적인 작업들을 하나로 묶는 헬퍼 함수
def sequential_transforms(*transforms):
  def callback(txt_input):
    for transform in transforms:
      txt_input = transform(txt_input)
    return txt_input
  return callback


# BOS/EOS를 추가하고 입력 순서(sequence) 인덱스에 대한 텐서를 생성하는 함수
def tensor_transform(token_ids: List[int]):
  return torch.cat((torch.tensor([SOS_TKN_IDX]),
                    torch.tensor(token_ids),
                    torch.tensor([EOS_TKN_IDX])))

# 출발어(src)와 도착어(tgt) 원시 문자열들을 텐서 인덱스로 변환하는 변형(transform)
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  text_transform[ln] = sequential_transforms(token_transform[ln], # 토큰화(Tokenization)
                                               vocab_transform[ln], # 수치화(Numericalization)
                                               tensor_transform) # BOS/EOS를 추가하고 텐서를 생성
```

앞에서 우리는 `token_transform`과 `vocab_transform` 등의 변환을 만들었습니다. 이것을 하나의 transform으로 묶어주기 위해 `text_transform`이라는 변환을 정의하겠습니다.

<br/>

```py
def greedy_decode(model, src: Tensor, src_mask: Tensor, max_len:int, start_symbol: int = SOS_TKN_IDX):
  src = src.to(DEVICE)
  src_mask = src_mask.to(DEVICE)

  memory = model.encode(src, src_mask) # context vector
  ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
  for i in range(max_len - 1):
    memory = memory.to(DEVICE)
    tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)

    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)

    prob = model.output_layer(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()

    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

    if next_word == EOS_TKN_IDX:
      break

  return ys


def translate(model, src_sentence: str):
  model.eval()
  
  src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
  num_tokens = src.shape[0]

  src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
  tgt_tokens = greedy_decode(model, src, src_mask, max_len = num_tokens + 5).flatten()
  tgt_tokens = list(tgt_tokens.cpu().numpy())

  return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(tgt_tokens)).replace("<sos>", "").replace("<eos>", "")

print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))
-----------------------------
A group of people stand in front of an igloo . 
```

다음으로 입력 받은 문장을 번역하는 `translate()` 함수를 정의하겠습니다. 이전의 `train_epoch()` 함수와 `evaluate()` 함수와 달리 Demo를 위해 사용할 함수입니다.

`greedy_decode()` 함수는 transformer 모델의 `forward()`가 아니라 Encoder만을 쓰는 `encode()`, Decoder만을 쓰는 `decoder()`로 나눠서 인코딩과 디코딩을 수행합니다.

최종 결과는 `"A group of people stand in front of an igloo . "`로 정답 문장인 `"A group of people are facing an igloo ."`와 매우 비슷하게 번역이 잘 되는 것을 볼 수 있습니다.

<br/>

```py
import random
val_iter = Multi30k(root='./data', split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

for idx, sample in enumerate(val_iter):
  if idx >= 5: break
  src_sentence = sample[0]
  gt_sentence = sample[1]
  output_sentence = translate(transformer, src_sentence)

  print(f'dutch: {src_sentence}')
  print(f'english(gt): {gt_sentence}')
  print(f'english(pred): {output_sentence}')
  print('-' * 50)
```

```text
dutch: Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen
english(gt): A group of men are loading cotton onto a truck
english(pred):  A group of men are loading into a truck of traffic . 
--------------------------------------------------
dutch:          Ein Mann schläft in einem grünen Raum auf einem Sofa.
english(gt):    A man sleeping in a green room on a couch.
english(pred):  A man is sleeping on a couch in a green room . 
--------------------------------------------------
dutch:          Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau.
english(gt):    A boy wearing headphones sits on a woman's shoulders.
english(pred):  A boy wearing headphones sits on his shoulders 's shoulders . 
--------------------------------------------------
dutch:          Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf
english(gt):    Two men setting up a blue ice fishing hut on an iced over lake
english(pred):  Two men are setting up a blue piece of plastic on a lake . 
--------------------------------------------------
dutch:          Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot.
english(gt):    A balding man wearing a red life jacket is sitting in a small boat.
english(pred):  A balding man is wearing a red life vest wearing a small life jacket . 
--------------------------------------------------
```

<hr/>

## 맺음말

이것으로 `nn.Transformer()` 모듈을 사용해 Trasnformer 번역 모델을 구현해보았습니다. Transformer는 이후에 BERT<small>(Bidirectional Encoder Representations from Transformers)</small>, GPT-3<small>(Generative Pre-Trained Transformer 3)</small>와 같이 최신 NLP 모델의 베이스가 되기 때문에 Transformer를 잘 이해하는 것이 중요합니다.

Transformer를 끝으로 PyTorch 세미나를 마무리 하겠습니다. 모두 수고하셨습니다!

<hr/>

## references

- ["NN.TRANSFORMER와 TORCHTEXT로 언어 번역하기"](https://tutorials.pytorch.kr/beginner/translation_transformer.html)