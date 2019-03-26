import gym

env = gym.make('CartPole-v0')
observations = env.reset()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('models/my-policy-model.meta')
    new_saver.restore(sess, 'models/my-policy-model')

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={x: observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])
env.close()