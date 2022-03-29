from os import stat
import numpy as np
from collections import deque
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


class DeepQNetwork:
    #state_num = 5, action_num = 3, final_state = 5
    r = np.array([[100, -1, -1, -1, -1, -1],
              [0, 100, 0, -1, -1, -1],
              [-1, 100, 0, -1, -1, -1],
              [-1, -1, 100, -1, -1, -1],
              [-1, 100, 0, -1, -1, -1],
              [-1, -1, -1, -1, 100, -1]
              ])
    '''
    r = np.array([[-1, -1, -1, -1, 0, -1],
                  [-1, -1, -1, 0, -1, 100.0],
                  [-1, -1, -1, 0, -1, -1],
                  [-1, 0, 0, -1, 0, -1],
                  [0, -1, -1, 1, -1, 100],
                  [-1, 0, -1, -1, 0, 100],
                  ])
    '''
    # 执行步数。
    step_index = 0

    # 状态数。
    state_num = 6

    # 动作数。
    action_num = 6

    # 训练之前观察多少步。
    OBSERVE = 1000.

    # 选取的小批量训练样本数。
    BATCH = 20

    # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
    FINAL_EPSILON = 0.0001

    # epsilon 的初始值，epsilon 逐渐减小。
    INITIAL_EPSILON = 0.1

    # epsilon 衰减的总步数。
    EXPLORE = 3000000.

    # 探索模式计数。
    epsilon = 0

    # 训练步数统计。
    learn_step_counter = 0

    # 学习率。
    learning_rate = 0.001

    # γ经验折损率。
    gamma = 0.9

    # 记忆上限。
    memory_size = 5000

    # 当前记忆数。
    memory_counter = 0

    # 保存观察到的执行过的行动的存储器，即：曾经经历过的记忆。
    replay_memory_store = deque()

    # 生成一个状态矩阵（6 X 6），每一行代表一个状态。
    state_list = None

    # 生成一个动作矩阵。
    action_list = None

    # q_eval 网络。
    q_eval_input = None
    action_input = None
    q_target = None
    q_eval = None
    predict = None
    loss = None
    train_op = None
    cost_his = None
    reward_action = None

    # tensorflow 会话。
    session = None

    def __init__(self, learning_rate=0.001, gamma=0.9, memory_size=5000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size

        # 初始化成一个 6 X 6 的状态矩阵。
        self.state_list = np.identity(self.state_num)

        # 初始化成一个 6 X 6 的动作矩阵。
        self.action_list = np.identity(self.action_num)

        # 创建神经网络。
        self.create_network()

        # 初始化 tensorflow 会话。
        self.session = tf.InteractiveSession()

        # 初始化 tensorflow 参数。
        self.session.run(tf.initialize_all_variables())

        # 记录所有 loss 变化。
        self.cost_his = []

    def create_network(self):
        """
        创建神经网络。
        :return:
        """
        self.q_eval_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, self.action_num], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)

        '''
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        '''
        neuro_layer_1 = 3
        w1 = tf.Variable(tf.random_normal([self.state_num, neuro_layer_1]))
        b1 = tf.Variable(tf.zeros([1, neuro_layer_1]) + 0.1)
        l1 = tf.nn.relu(tf.matmul(self.q_eval_input, w1) + b1)

        w2 = tf.Variable(tf.random_normal([neuro_layer_1, self.action_num]))
        b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        self.q_eval = tf.matmul(l1, w2) + b2

        # 取出当前动作的得分。
        self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square((self.q_target - self.reward_action)))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.predict = tf.argmax(self.q_eval, 1)

    def select_action(self, state_index):
        """
        根据策略选择动作。
        :param state_index: 当前状态。
        :return:
        """
        current_state = self.state_list[state_index:state_index + 1]

        if np.random.uniform() < self.epsilon:
            current_action_index = np.random.randint(0, self.action_num)
        else:
            actions_value = self.session.run(self.q_eval, feed_dict={self.q_eval_input: current_state})
            action = np.argmax(actions_value)
            current_action_index = action

        # 开始训练后，在 epsilon 小于一定的值之前，将逐步减小 epsilon。
        if self.step_index > self.OBSERVE and self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

        return current_action_index

    def step(self, state, action):
        """
        执行动作。
        :param state: 当前状态。
        :param action: 执行的动作。
        :return:
        """
        reward = self.r[state][action]
        
        '''動作結束後直接結束進入下一次的訓練
        next_state = 3
        '''
        next_state = 5

        done = False

        ''' end = 3:結束動作
        if action == 3:
            done = True
        '''
        if next_state == 5:
            done = True

        return next_state, reward, done

    def save_store(self, current_state_index, current_action_index, current_reward, next_state_index, done):
        """
        保存记忆。
        :param current_state_index: 当前状态 index。
        :param current_action_index: 动作 index。
        :param current_reward: 奖励。
        :param next_state_index: 下一个状态 index。
        :param done: 是否结束。
        :return:
        """
        #current_state_index = current_state(在隨機訓練時隨機跑得狀態 )
        #next_state_index = next_state(從step跑)
        current_state = self.state_list[current_state_index:current_state_index + 1]
        current_action = self.action_list[current_action_index:current_action_index + 1]
        next_state = self.state_list[next_state_index:next_state_index + 1]
        # 记忆动作(当前状态， 当前执行的动作， 当前动作的得分，下一个状态)。
        self.replay_memory_store.append((
            current_state,
            current_action,
            current_reward,
            next_state,
            done))

        # 如果超过记忆的容量，则将最久远的记忆移除。
        if len(self.replay_memory_store) > self.memory_size:
            self.replay_memory_store.popleft()

        self.memory_counter += 1

    def experience_replay(self):
        """
        记忆回放。
        :return:
        """
        # 随机选择一小批记忆样本。
        batch = self.BATCH if self.memory_counter > self.BATCH else self.memory_counter
        #從replay_memory_store裡取出batch數量的記憶樣本來做計算
        minibatch = random.sample(self.replay_memory_store, batch)

        batch_state = None
        batch_action = None
        batch_reward = None
        batch_next_state = None
        batch_done = None

        for index in range(len(minibatch)):
            if batch_state is None:
                batch_state = minibatch[index][0]
            elif batch_state is not None:
                batch_state = np.vstack((batch_state, minibatch[index][0]))

            if batch_action is None:
                batch_action = minibatch[index][1]
            elif batch_action is not None:
                batch_action = np.vstack((batch_action, minibatch[index][1]))

            if batch_reward is None:
                batch_reward = minibatch[index][2]
            elif batch_reward is not None:
                batch_reward = np.vstack((batch_reward, minibatch[index][2]))

            if batch_next_state is None:
                batch_next_state = minibatch[index][3]
            elif batch_next_state is not None:
                batch_next_state = np.vstack((batch_next_state, minibatch[index][3]))

            if batch_done is None:
                batch_done = minibatch[index][4]
            elif batch_done is not None:
                batch_done = np.vstack((batch_done, minibatch[index][4]))

        # q_next：下一个状态的 Q 值。
        q_next = self.session.run([self.q_eval], feed_dict={self.q_eval_input: batch_next_state})

        q_target = []
        for i in range(len(minibatch)):
            # 当前即时得分。
            current_reward = batch_reward[i][0]

            # # 游戏是否结束。
            # current_done = batch_done[i][0]

            # 更新 Q 值。
            q_value = current_reward + self.gamma * np.max(q_next[0][i])

            # 当得分小于 0 时，表示走了不可走的位置。
            if current_reward < 0:
                q_target.append(current_reward)
            else:
                q_target.append(q_value)

        _, cost, reward = self.session.run([self.train_op, self.loss, self.reward_action],
                                           feed_dict={self.q_eval_input: batch_state,
                                                      self.action_input: batch_action,
                                                      self.q_target: q_target})

        self.cost_his.append(cost)

        # if self.step_index % 1000 == 0:
        #     print("loss:", cost)

        self.learn_step_counter += 1

    def train(self):
        """
        训练。
        :return:
        """
        # 初始化当前状态。
        #randint(start, end)
        current_state = np.random.randint(0, self.state_num - 1)
        self.epsilon = self.INITIAL_EPSILON

        while True:
            # 选择动作。
            action = self.select_action(current_state)

            # 执行动作，得到：下一个状态，执行动作的得分，是否结束。
            next_state, reward, done = self.step(current_state, action)

            # 保存记忆。
            self.save_store(current_state, action, reward, next_state, done)

            # 先观察一段时间累积足够的记忆在进行训练。
            #step_index = self.memory_counter = 訓練次數
            if self.step_index > self.OBSERVE:
                self.experience_replay()

            if self.step_index > 10000:
                break

            #判斷是否結束，若是結束重製start_state，否則current_state = next_state
            if done:
                current_state = np.random.randint(0, self.action_num - 1)
            else:
                current_state = next_state

            self.step_index += 1
            
    def pay(self):
        self.train()
        '''輸出Q-table'''
        for index in range(5):
            print(index)
            start_room = index
            current_state = start_room
            out_result = self.session.run(self.q_eval, feed_dict={self.q_eval_input: self.state_list[current_state:current_state + 1]})
            print(out_result)
        
        for index in range(5):
            start_state = index
            if start_state == 0:
                print("start-state: Target standing in the distance and waving hand")
            elif start_state == 1:
                print("start-state: Target standing in the distance ")
            elif start_state == 2:
                print("start-state: Target approaching ")
            elif start_state ==3:
                print("start-state: Target stand in front of the robot ")
            elif start_state ==4:
                print("start-state: Target passing ")

            #print("start-state:",start_state, "#############################")
            current_state = start_state

            step = 0
            target_state = 10
            
            while current_state != target_state:

                out_result = self.session.run(self.q_eval, feed_dict={self.q_eval_input: self.state_list[current_state:current_state + 1]})

                action = np.argmax(out_result)
                if action == 0:
                    print("state", current_state, "Robot need to closing target 0")
                elif action == 1:
                    print("state", current_state, "Robot need to stop 1")
                elif action == 2:
                    print("state", current_state, "Robot need to ask question 2")

                #print("state", current_state, "action", action)

                current_state = 10
                step += 1
            
            print("#############################")
            

            


if __name__ == "__main__":
    q_network = DeepQNetwork()
    q_network.pay()

