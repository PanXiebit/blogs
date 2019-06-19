# TensorArray 和 基于lstm的MDP模拟文本生成
这也是seqgan的核心，用Monte Carlo search代替sampling来选择next token.在看具体代码之前先了解下 tensorarray.

## TensorArray
> Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays
This class is meant to be used with dynamic iteration primitives such as while_loop and map_fn. It supports gradient back-propagation via special "flow" control flow dependencies.
一个封装了动态大小、per-time-step 写入一次的 tensor数组的类。在序列生成中，序列的长度通常是不定的，所以会需要使用动态tensorarray.

### 类初始化
```python
def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):
```

- size: int32 scalar `Tensor`, 动态数组的大小
- dynamic_size: Python bool, 是否可以增长，默认false

### 方法
- stack
```Python
def stack(self, name=None):
  """Return the values in the TensorArray as a stacked `Tensor`.
  """
```
将动态数组 stack 起来，得到最终的 tensor.

- concat
```python
def concat(self, name=None):
  """Return the values in the TensorArray as a concatenated `Tensor`.
  """
```
将动态数组 concat 起来，得到最终的 tensor.

- read  
```python
def read(self, index, name=None):
  """Read the value at location `index` in the TensorArray.
  读过一次之后会清0. 不能读第二次。但可以再次写入之后。
  """
```
- write  
```python  
def write(self, index, value, name=None):
  """Write `value` into index `index` of the TensorArray.
  """
  - index: int32 scalar with the index to write to.
  - value: ttf.Tensor
```

- gather  
- unstack  
- split  
- scatter  

### tf.while_loop
```python  
def while_loop_v2(cond,
                  body,
                  loop_vars,
                  shape_invariants=None,
                  parallel_iterations=10,
                  back_prop=True,
                  swap_memory=False,
                  maximum_iterations=None,
                  name=None):
"""Repeat `body` while the condition `cond` is true.
"""
- cond: callable, return boolean scalar tensor. 参数个数必须和 loop_vars 一致。  
- body: vallable. 循环执行体，参数个数必须和 loop_vars 一致.
- loop_vars: 循环变量，tuple, namedtuple or list of numpy array.
```

### example:

```python
matrix = tf.random.normal(shape=[5, 1], dtype=tf.float32)
sequence_length = 5
gen_o = tf.TensorArray(dtype=tf.float32, size=sequence_length,
                       dynamic_size=False, infer_shape=True)
init_state = (0, gen_o)
condition = lambda i, _: i < sequence_length
body = lambda i, gen_o : (i+1, gen_o.write(i, matrix[i] * 2))
n, gen_o = tf.while_loop(condition, body, init_state)
gen_o_stack = gen_o.stack()
gen_o_concat = gen_o.concat()用 LSTM 模拟马尔科夫决策过程

print(gen_o)                     # TensorArray object
print(gen_o_stack)               # tf.Tensor(), [5,]
print(gen_o_concat)              # tf.Tensor(), [5,1]
print(gen_o.read(3))             # -0.22972003, tf.Tensor  读过一次就被清0了
print(gen_o.write(3, tf.constant([0.22], dtype=tf.float32)))  # TensorArray object
print(gen_o.concat())            # tf.Tensor([-2.568663 0.09471891 1.2042408 0.22 0.2832177 ], shape=(5,), dtype=float32)
print(gen_o.read(3))             # tf.Tensor([0.22], shape=(1,), dtype=float32)
print(gen_o.read(3))             # Could not read index 3 twice because it was cleared after a previous read
```

## 用 LSTM 模拟马尔科夫决策过程

- current time t state: $(y_1,...,y_t)$. 但是马尔科夫决策过程的原理告诉我们**一旦当前状态确定后，所有的历史信息都可以扔掉了。这个状态足够去预测 future.** 所以在LSTM里面就是隐藏状态 $h_{t-1}$. 以及当前可观测信息 $x_t$.  
- action a: 选择 next token $y_t$.
- policy: $G_{\theta}(y_t|Y_{1:t-1})$. 也就是生成next token的策略。下面代码的方法 $o_t \rightarrow log(softmax(o_t))$. 然后基于这个 log-prob 的分布进行 sample. 问题是这个过程可导吗？？？  
-


```python  
# initial states
self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
self.h0 = tf.stack([self.h0, self.h0])

# generator on initial randomness
gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                     dynamic_size=False, infer_shape=True)
gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                     dynamic_size=False, infer_shape=True)

# RL process
def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
    h_t = self.g_recurrent_unit(x_t, h_tm1)  # lstm(x_t, h_{t-1}) ->h_t. [batch. hidden_size * 2], hidden_memory_tuple
    o_t = self.g_output_unit(h_t)  # [batch, vocab_size] , logits not prob
    log_prob = tf.math.log(tf.nn.softmax(o_t))  # log-prob  # [batch, vocab_size]

    # Monte Carlo search? 多项分布 Multinomial
    self.token_search = multinomial.Multinomial(total_count=self.batch_size, logits=log_prob)
    next_token = tf.argmax(self.token_search.probs, axis=-1) # [batch]
    # next_token = tf.cast(tf.reshape(multinomial.Multinomial(log_prob, 1), [self.batch_size]), tf.int32)  # [batch]

    x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # x_{t+1}, [batch, emb_dim]
    gen_o = gen_o.write(index=i,
                        value=tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                        tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
    gen_x = gen_x.write(i, next_token)  # indices, batch_size
    return i + 1, x_tp1, h_t, gen_o, gen_x

n, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
    cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
    body=_g_recurrence,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x)
    )
assert n == self.sequence_length

self.gen_x = self.gen_x.stack()  # seq_length x batch_size
self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length
```
