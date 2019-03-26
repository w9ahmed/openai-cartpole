import tensorflow as tf
import numpy as np
import gym

num_inputs = 4
num_hidden = 4
num_outputs = 1 


learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

x = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden_layer = tf.layers.dense(x, num_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden_layer, num_outputs)
outputs = tf.nn.sigmoid(logits)


probabilities = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(probabilities, num_samples=1)

y = 1.0  - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gradients_and_variables = optimizer.compute_gradients(cross_entropy)


gradients = []
gradient_placeholders = []
grads_and_vars_feed = []


for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def helper_discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discounter_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards, discount_rate))
    
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


env = gym.make('CartPole-v0')
num_game_rounds = 10
max_game_steps = 1000
num_iterations = 750
discount_rate = 0.9

""" with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_iterations):
        print('On iteration %d' % iteration)

        all_rewards = []
        all_gradients = []

        for game in range(num_game_rounds):
            current_rewards = []
            current_gradients = []

            observations = env.reset()
            for step in range(max_game_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={x: observations.reshape(1, num_inputs)})

                observations, reward, done, info = env.step(action_val[0][0])

                current_rewards.append(reward)
                current_gradients.append(gradients_val)

                if done:
                    break
        
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
    
        all_rewards = discounter_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}

        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index] for game_index, rewards in enumerate(all_rewards) for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        
        sess.run(training_op, feed_dict=feed_dict)
        print('Saving graph and session')
        meta_graph_def = tf.train.export_meta_graph(filename='models/my-policy-model.meta')
        saver.save(sess, 'models/my-policy-model') """




env = gym.make('CartPole-v0')
observations = env.reset()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('models/my-policy-model.meta')
    new_saver.restore(sess, 'models/my-policy-model')

    for _ in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={x: observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])
env.close()