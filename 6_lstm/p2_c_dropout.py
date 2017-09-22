from helper_functions import *

num_nodes = 64
embedding_size = 128

# Define the Graph.

graph = tf.Graph()
with graph.as_default():
  # Now vocab_embedding shape is [27*27, embedding_size]
  vocab_embeddings = tf.Variable(tf.random_uniform([vocabulary_size*vocabulary_size, embedding_size], -1.0, 1.0))
  # Parameters:
  # Input gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
  cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  # Better to look at this -> http://colah.github.io/posts/2015-08-Understanding-LSTMs/
  def lstm_cell(i, o, state):
    dropped_i = tf.nn.dropout(i, 0.5)
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb 
    state = forget_gate * state + input_gate * tf.tanh(update) # to keep things between -1 and +1 
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state

  # Input data is a list with eleven (64 by 27) tensors
  train_data = list() # 'ons anarchi': 'ists advoca', 'when milita': 'ary governm'
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))

  train_chars = train_data[:num_unrollings]  
  train_inputs = zip(train_chars[:-1], train_chars[1:])
  train_labels = train_data[2:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
    
  for i in train_inputs:
    # Now train_inputs is a tuple. 
    bigram_index= tf.argmax(i[0], axis=1) * vocabulary_size + tf.argmax(i[1], axis=1)
    i_embedding = tf.nn.embedding_lookup(vocab_embeddings, bigram_index)
    output, state = lstm_cell(i_embedding, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.concat(train_labels, 0), logits=logits))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = [] # in Bigram implementation, sample_input becomes a list!
  # sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size]) <- for unigram
  for _ in range(2):
    sample_input.append(tf.placeholder(tf.float32, shape=[1, vocabulary_size]))
  
  sample_embedding_input = tf.nn.embedding_lookup(vocab_embeddings, tf.argmax(sample_input[0], axis=1)* vocabulary_size + tf.argmax(sample_input[1], axis=1))
  
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))

  sample_output, sample_state = lstm_cell(
    sample_embedding_input, saved_sample_output, saved_sample_state)
  
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))



import collections
num_steps = 7001
summary_frequency = 100

valid_batches = BatchGenerator(valid_text, 1, 2) #(text, batch_size, num_unrollings)



# Session starts!

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next() # But then .next() will be called only 7001 times!
    """ |
        v
    train_text:  ons anarchists advocate social relations based upon 
    train_batches no. 1
    ['ons anarchi', 'when milita', 'lleria arch']
    train_batches no. 2
    ['ists advoca', 'ary governm', 'hes nationa'] <- batches
    train_batches no. 3
    ['ate social ', 'ments faile', 'al park pho']
    """
    feed_dict = dict() # contain the set of examples on which to train for the step, keyed by the placeholder ops they represent.
    
    for i in range(num_unrollings + 1): # train_data: [(64,27), (64,27), (64,27), (64,27), (64,27) ...]
      feed_dict[train_data[i]] = batches[i] # 'ons anarchi': 'ists advoca', 'when milita': 'ary governm'
    
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    
    # Done.
    
    #####################################################################
    # Summary
    #####################################################################
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[2:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80) # ============================================================
        for _ in range(5):
          #feed = sample(random_distribution())
          feed = collections.deque(maxlen=2)
          for _ in range(2):  
            feed.append(random_distribution())
          sentence = characters(feed[0])[0] + characters(feed[1])[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input[0]: feed[0], sample_input[1]: feed[1]})
            #feed = sample(prediction)
            feed.append(sample(prediction))
            #sentence += characters(feed)[0]
            sentence += characters(feed[1])[0]
          print(sentence)
        print('=' * 80) # ============================================================
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0    
      for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input[0]: b[0], sample_input[1]: b[1]})
        valid_logprob = valid_logprob + logprob(predictions, b[2])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))