import gym

env = gym.make('CartPole-v0')

observation = env.reset()
print('-> Initial Observation: %s <-\n' % observation)

for _ in range(1000):
    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    action = 0
    if pole_ang > 0:
        action = 1
    
    print('- Step %d: ' % _)
    observation, reward, done, info = env.step(action)
    print('\tAction: %s' % action)
    print('\tObservation: %s' % observation)
    print('\tReward: %s' % reward)
    print('\tDone: %s' % done)
    print('\tInfo: %s' % info)
    
env.close()