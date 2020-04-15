import tensorflow as tf
import numpy as np
from collections import deque
import random

FRAME_PER_ACTION = 1  # 几帧图像进行一次动作
OBSERVE = 100.  # 观察期的长度
EXPLORE = 200000.  # 探索期长度
REPLAY_MEMORY = 50000  # 设置REPLAY_MEMORY容量
BATCH_SIZE = 32  # 每次更新网络参数时使用的样本数目

GAMMA = 0.99  # 折扣系数
FINAL_EPSILON = 0.0001  # 最终EPSILON大小
INITIAL_EPSILON = 0.01  # 训练起始时，EPSILON的大小

UPDATE_TIME = 100

try:
    tf.mul
except:
    # 兼容新版本tensorflow
    tf.mul = tf.multiply


class RLBrain:

    def __init__(self, actions):
        # 使用双向列表来存放replayMemory
        self.replayMemory = deque()
        # 初始化参数
        self.time = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # 初始化DQN
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.DQN_Create()

        # 初始化Target DQN
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.DQN_Create()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.gredient()

        # 保存和读取训练的网络
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("加载成功:", checkpoint.model_checkpoint_path)
        else:
            print("加载失败")

    def DQN_Create(self):
        # 构建网络权重
        # 使用32个大小为8x8x4大小的卷积核，卷积得到20x20x32大小的矩阵
        W_conv1 = self.define_weight([8, 8, 4, 32])
        b_conv1 = self.define_variable([32])
        W_conv2 = self.define_weight([4, 4, 32, 64])
        b_conv2 = self.define_variable([64])
        W_conv3 = self.define_weight([3, 3, 64, 64])
        b_conv3 = self.define_variable([64])
        W_fc1 = self.define_weight([1600, 512])
        b_fc1 = self.define_variable([512])
        W_fc2 = self.define_weight([512, self.actions])
        b_fc2 = self.define_variable([self.actions])

        # 80x80大小的4帧图像
        stateInput = tf.placeholder("float", [None, 80, 80, 4])

        # 隐藏层
        # conv2d参数为（输入，卷积核，步长）
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
        # 需要对以上矩阵进行不重叠的池化操作，池化窗口大小为2x2
        h_pool1 = self.max_pool(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

    def TDQN_Copy(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def gredient(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def DQN_train(self):

        random_batch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in random_batch]
        action_batch = [data[1] for data in random_batch]
        reward_batch = [data[2] for data in random_batch]
        nextState_batch = [data[3] for data in random_batch]

        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            terminal = random_batch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # 保存网络权重
        if self.time % 50000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'DQN' + '-time', global_step=self.time)

        if self.time % UPDATE_TIME == 0:
            self.TDQN_Copy()

    def getStarted(self, next_obs, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        new_state = np.append(self.currentState[:, :, 1:], next_obs, axis=2)
        self.replayMemory.append((self.currentState, action, reward, new_state, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.time > OBSERVE:
            self.DQN_train()

        # 以下在控制台打印
        state = ""
        if self.time <= OBSERVE:
            state = "observe"
        elif OBSERVE < self.time <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("CURRENT_TIME: ", self.time, " ------ CURRENT_STATE: ", state, " ------ CURRENT_EPSILON", self.epsilon)

        self.currentState = new_state
        self.time += 1

    def choose_action(self):
        q_value = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if self.time % FRAME_PER_ACTION != 0:
            action[0] = 1  # 无动作
        else:
            # epsilon-greedy算法
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(q_value)
                action[action_index] = 1

        # 在观察期后不断修改epsilon
        if self.time > OBSERVE and self.epsilon > FINAL_EPSILON:
            abs_deta = INITIAL_EPSILON - FINAL_EPSILON
            deta = abs_deta / EXPLORE
            self.epsilon -= deta

        return action

    def InitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    # 定义卷积操作，实现卷积核W在数据x上卷积操作
    # strides为卷积核的移动步长， padding=same使得输入和输出图像大小相同
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    # 定义池化函数，调用max_pool执行最大池化操作
    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def define_weight(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def define_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)