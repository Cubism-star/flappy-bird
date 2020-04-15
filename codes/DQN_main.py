import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from DQN_Brain import RLBrain
import numpy as np

# 预处理图像：
# 1.转化为80x80大小
# 2.转化为灰度图
# 3.把图像二值化为黑白两色，即1或者255
# 4.将图像转化为80x80x1的矩阵
#
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	# threshold(src, thresh, maxval, type[, dst])用于设定阈值
	ret, observation = cv2.threshold(observation,1,255, cv2.THRESH_BINARY)
	return np.reshape(observation, (80, 80, 1))

def playGame():
	# 动作数目，跳越或无动作
	actions = 2
	brain = RLBrain(actions)
	# 创建游戏
	flappyBird = game.GameState()

	# 构造初始状态
	init_action = np.array([1,0])
	init_observation, init_reward, terminal = flappyBird.frame_step(init_action)
	init_observation = cv2.cvtColor(cv2.resize(init_observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, init_observation = cv2.threshold(init_observation, 1, 255, cv2.THRESH_BINARY)
	brain.InitState(init_observation)

	# 循环训练
	while True:
		action = brain.choose_action()
		nextObservation, reward, terminal = flappyBird.frame_step(action)
		nextObservation = preprocess(nextObservation)
		brain.getStarted(nextObservation, action, reward, terminal)

def main():
	playGame()

if __name__ == '__main__':
	main()