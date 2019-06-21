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
  - value: tf.Tensor
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
- policy: $G_{\theta}(y_t|Y_{1:t-1})$. 也就是生成next token的策略。下面代码的方法 $o_t \rightarrow log(softmax(o_t))$. 然后基于这个 log-prob 的分布进行 sample. 问题是这个过程不可导呀？  


### generator
这是生成器生成sample的过程，初始状态是 $h_0$.

g_recurrence 就是step-by-step的过程，next_token是通过tf.multinomial采样得到的，其采样的distribution是 log_prob [tf.log(tf.nn.softmax(o_t))]。

```python  
class Generator(tf.keras.Model):
  ...

  self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
  self.h0 = tf.stack([self.h0, self.h0])

  # define variables
  self.g_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))
  self.g_params.append(self.g_embeddings)
  self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_{t-1} to h_t for generator
  self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

  def _unsuper_generate(self):
      """ unsupervised generate. using in rollout policy.
      :return: 生成得到的 token index
      """
      """
      :param input_x:  [batch, seq_len]
      :param rewards:  [batch, seq_len]
      :return:
      """
      gen_o = tf.TensorArray(dtype=tf.float32, size=self.sequence_length,
                             dynamic_size=False, infer_shape=True)
      gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                             dynamic_size=False, infer_shape=True)

      def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
          h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
          o_t = self.g_output_unit(h_t)  # [batch, vocab] , logits not prob
          log_prob = tf.log(tf.nn.softmax(o_t))
          #tf.logging.info("unsupervised generated log_prob:{}".format(log_prob[0]))
          next_token = tf.cast(tf.reshape(tf.multinomial(logits=log_prob, num_samples=1),
                                          [self.batch_size]), tf.int32)
          x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # [batch, emb_dim]
          gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                           tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
          gen_x = gen_x.write(i, next_token)  # indices, batch_size
          return i + 1, x_tp1, h_t, gen_o, gen_x

      _, _, _,  def _super_generate(self, input_x):
      """ supervised generate.

      :param input_x:
      :return: 生成得到的是 probability [batch * seq_len, vocab_size]
      """
      with tf.device("/cpu:0"):
          self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, input_x),
                                          perm=[1, 0, 2])  # [seq_len, batch_size, emb_dim]
      # supervised pretraining for generator
      g_predictions = tf.TensorArray(
          dtype=tf.float32, size=self.sequence_length,
          dynamic_size=False, infer_shape=True)

      ta_emb_x = tf.TensorArray(
          dtype=tf.float32, size=self.sequence_length)
      ta_emb_x = ta_emb_x.unstack(self.processed_x) self.gen_o, self.gen_x = tf.while_loop(
          cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
          body=_g_recurrence,
          loop_vars=(tf.constant(0, dtype=tf.int32),
                     tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                     self.h0, gen_o, gen_x))

      self.gen_x = self.gen_x.stack()  # [seq_length, batch_size]
      self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # [batch_size, seq_length]
      return self.gen_x
```

所以是通过monte carlo的形式生成fake sample，作为discriminator的输入吗？那这个过程也不可导呀。其实不是这样的。我们再看对抗学习中更新generator的代码:

```python
def gen_reward_train_step(x_batch, rewards):
    with tf.GradientTape() as tape:
        g_loss = generator._get_generate_loss(x_batch, rewards)
        g_gradients, _ = tf.clip_by_global_norm(
            tape.gradient(g_loss, generator.trainable_variables), clip_norm=5.0)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    return g_loss

tf.logging.info("------------------ 6. start Adversarial Training...--------------------------")
for total_batch in range(TOTAL_BATCH):
    # fix discriminator, and train the generator for one step
    for it in range(1):
        samples = generator._unsuper_generate()
        #tf.logging.info("unsuper generated samples:{}".format(samples[0]))
        rewards = rollout.get_reward(samples, rollout_num=2, discriminator=discriminator)  # 基于 monte carlo 采样16，计算并累计 reward.
        #tf.logging.info("reward:{}".format(rewards[0]))
        gen_reward_train_step(samples, rewards)        # update generator.
        # Update roll-out parameters
    rollout.update_params()   # update roll-out policy.
```

这儿采用的是 `generator._get_generate_loss`， 所以它对generator的参数都是可导的吗？ 我们再看这个生成器中这个function的代码：

```python
class Generator(tf.keras.Model):
  ...

  def _super_generate(self, input_x):
      """ supervised generate.

      :param input_x:
      :return: 生成得到的是 probability [batch * seq_len, vocab_size]
      """
      with tf.device("/cpu:0"):
          self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, input_x),
                                          perm=[1, 0, 2])  # [seq_len, batch_size, emb_dim]
      # supervised pretraining for generator
      g_predictions = tf.TensorArray(
          dtype=tf.float32, size=self.sequence_length,
          dynamic_size=False, infer_shape=True)

      ta_emb_x = tf.TensorArray(
          dtype=tf.float32, size=self.sequence_length)
      ta_emb_x = ta_emb_x.unstack(self.processed_x)

    def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
        h_t = self.g_recurrent_unit(x_t, h_tm1)
        o_t = self.g_output_unit(h_t)
        g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # [batch, vocab_size]
        x_tp1 = ta_emb_x.read(i)                                    # supervised learning, teaching forcing.
        return i + 1, x_tp1, h_t, g_predictions

    _, _, _, self.g_predictions = tf.while_loop(
        cond=lambda i, _1, _2, _3: i < self.sequence_length,
        body=_pretrain_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                   tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                   self.h0, g_predictions))
    self.g_predictions = tf.transpose(self.g_predictions.stack(),
                                      perm=[1, 0, 2])  # [batch_size, seq_length, vocab_size]
    self.g_predictions = tf.clip_by_value(
        tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)  # [batch_size*seq_length, vocab_size]
    return self.g_predictions       # [batch_size*seq_length, vocab_size]

    def _get_generate_loss(self, input_x, rewards):
        """

        :param input_x: [batch, seq_len]
        :param rewards: [batch, seq_len]
        :return:
        """
        self.g_predictions = self._super_generate(input_x)
        real_target = tf.one_hot(
            tf.to_int32(tf.reshape(input_x, [-1])),
            depth=self.vocab_size, on_value=1.0, off_value=0.0)  # [batch_size * seq_length, vocab_size]
        self.pretrain_loss = tf.nn.softmax_cross_entropy_with_logits(labels=real_target,
                                                                     logits=self.g_predictions)  # [batch * seq_length]
        self.g_loss = tf.reduce_mean(self.pretrain_loss * tf.reshape(rewards, [-1]))  # scalar
        return self.g_loss
```

所以seqgan的作者是怎么做的呢，利用 `generator._unsuper_generate`先生成fake sample，然后再利用 `generator._super_generate` 得到 `g_predictions`, 将fake sample作为 `real_target` 与 `g_predictions` 做交叉熵求出 `pretrain_loss`，然后乘以每一个token对应的rewards得到最终的loss. 这个过程是可导的。

所以先用预训练好的 generator 生成samples，作为tagrget（这个过程不可导，）。再利用 generator 有监督的生成 samples,作为 prediction(这个过程可导).然后计算 prediction_loss（类似于传统的training）.只不过这里还乘上了利用 roll-out policy 得到的rewards.使得每一个词的权重都是不一样的。

### roll-policy
这个过程比较容易理解，对于给定的 given_num,小于 given_num 的直接 copy，但是 $h_t$ 的计算依旧。大于 given_num 的token采用 `generate._unsuper_generate`.
