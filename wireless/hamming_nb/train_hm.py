# This code is developed based on RL code in the following repository in github
# https://github.com/rlcode/reinforcement-learning-kr-v2
import copy
import matplotlib.pyplot as plt
import random
import numpy as np
from environment_hm import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam

# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy


# 그리드월드 예제에서의 REINFORCE 에이전트
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        
        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = REINFORCE(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        # discounted_rewards -= np.mean(discounted_rewards)
        # discounted_rewards /= np.std(discounted_rewards)
        
        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = -policies * tf.math.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)


def policy_to_action(policy, std):
    """
    - 아래에서 TH=0.5가 된 이유는 결과값이 0, 1이 되기 때문에 그 중간을 사용한 것이다. 
      action[action_samples > 0.5] = 1
    - Gaussian policy의 오류 값인 delta = actions - policies가 제대로 구해지게 한다.
      결국 action이 0이 되게 하기 위해서는 policy가 0가 나오고,
      1이 되게 하기 위해서는 policy가 1이 나오게 하는게 목적임.
    - 둘이 비슷해지면 최적화가 된 것이다. 
    - 둘이 다를 경우(0.1 exploitation)에 해당 action으로 인해 G가 크게 나왔다면 그 부분은 특히 수정하도로 함.
    """
    mu_array = np.array(policy)
    action_samples = np.array([np.random.normal(mu,std) for mu in mu_array])
    action = np.zeros(len(action_samples), dtype=int)
    action[action_samples > 0.5] = 1
    # action[action_samples > TH] = 1
    return action

def policies_to_probs_actions(policies, std):
    """
    - 아래에서 TH=0.5가 된 이유는 결과값이 0, 1이 되기 때문에 그 중간을 사용한 것이다. 
      action[action_samples > 0.5] = 1
    - Gaussian policy의 오류 값인 delta = actions - policies가 제대로 구해지게 한다.
      결국 action이 0이 되게 하기 위해서는 policy가 0가 나오고,
      1이 되게 하기 위해서는 policy가 1이 나오게 하는게 목적임.
    - 둘이 비슷해지면 최적화가 된 것이다. 
    - 둘이 다를 경우(0.1 exploitation)에 해당 action으로 인해 G가 크게 나왔다면 그 부분은 특히 수정하도로 함.
    """
    mu_array = np.array(policies)
    probs = np.array([np.random.normal(mu,std) for mu in mu_array])
    actions = np.zeros(len(probs), dtype=int)
    actions[probs > 0.5] = 1
    return probs, actions

# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE_Gaussian(tf.keras.Model):
    def __init__(self, action_size, N_code:int=7, K_code:int=4):
        super().__init__()
        self.fc1 = Dense(2*K_code*(N_code-K_code))
        self.fc2 = Dense(2*K_code*(N_code-K_code))
        self.fc_out = Dense(action_size) # No softmax since Gaussian agent

    def call(self, x):
        x = self.fc1(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
        x = self.fc2(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.1)(x)
        policy = self.fc_out(x)
        return policy

class REINFORCEAgentGaussian(REINFORCEAgent):
    def __init__(self, state_size, action_size, N_code:int=7, K_code:int=4):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        self.std = np.sqrt(0.1)
        
        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = REINFORCE_Gaussian(self.action_size, N_code, K_code)
        self.optimizer = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.model(state)[0]
        #mu_array = np.array(policy)
        #action_samples = np.array([np.random.normal(mu,self.std) for mu in mu_array])
        #action = np.zeros(len(action_samples), dtype=int)
        #action[action_samples > 0] = 1
        return policy_to_action(policy, self.std)

    def train_model(self):
        # https://hatter30.github.io/blog/describe-policy-gradient/
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        #print(self.rewards, discounted_rewards) 
        
        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies_tf = self.model(np.array(self.states))
            #print('policies= ', policies.numpy())
            probs_array = []
            actions_array = []
            for policies in policies_tf:
                probs, actions = policies_to_probs_actions(policies, self.std)
                probs_array.append(probs)
                actions_array.append(actions)
            probs_array = np.array(probs_array)
            actions_array = np.array(actions_array)
            delta = probs_array - policies_tf
            loss = -tf.reduce_sum(delta * discounted_rewards)
            entropy = - policies * tf.math.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)

if __name__ == "__main__":
    # 환경과 에이전트 생성
    # env = Env(render_speed=0.01)
    Target_BLER = 0.05
    N_code = 7
    K_code = 4
    env = Env(N_code, K_code, Target_BLER)
    state_size = env.state_size
    action_size = env.action_size

    agent = REINFORCEAgentGaussian(state_size, action_size, N_code, K_code)

    scores, episodes = [], []

    EPISODES = 100
    plt.axis([0,EPISODES,-10,0])
    plt.ion() # interactive mode를 on한다. ioff()는 off한다. cla()는 화면을 지운다.
    plt.show()
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.append_sample(state, action, reward)
            score += reward

            state = next_state

            if done:
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d}".format(e, score))

                scores.append(score)
                episodes.append(e)

        # 100 에피소드마다 모델 저장        
        if e % 10 == 0:
            print('action =', action)
            agent.model.save_weights('save_model/model', save_format='tf')
        
            plt.axis([0,EPISODES,-10,0])
            plt.plot(scores, '-b')
            plt.draw()
            plt.pause(0.001)    
            plt.cla()

    plt.ioff()
    plt.axis([0,EPISODES,-10,0])
    plt.plot(scores, '-b')
    plt.xlabel('Epoch')
    plt.ylabel('Required SNR(dB)')
    plt.title('PG RL Search for the Best (7,4) code')
    plt.show()