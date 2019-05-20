import numpy as np

class Gomoku():
    def __init__(self,size):
        #player1先走
        self.chessboard=np.zeros((size,size)) #初始化棋盘，玩家1填1，玩家2填2
        self.lastTimeStep=2 #上一步是哪个玩家走的
        self.size=size
    def step(self,player,action):
        if player == 1:
            if self.lastTimeStep == 2:
                self.lastTimeStep=1
                done=self.checkFoul(1,action) #在已落子格子上落子
                if done:
                    # print("玩家1犯规，游戏重启")
                    self.lastTimeStep = 2 #游戏重启，玩家1先走
                    self.chessboard = np.zeros((self.size, self.size)) #从新初始化棋盘
                    next_observation = self.chessboard
                    reward = -0.01
                    done = True
                    info = {}
                    return next_observation,reward,done,info
                else:
                    win=self.checkWin(player)
                    if win:
                        # print("玩家1赢！,游戏重置")
                        self.lastTimeStep = 2  # 游戏重启，玩家1先走
                        self.chessboard = np.zeros((self.size, self.size))  # 从新初始化棋盘
                        next_observation = self.chessboard
                        reward = 1.0
                        done = True
                        info = {}
                        return next_observation, reward, done, info
                    else:
                        draw=self.checkDraw()
                        if draw:
                            self.chessboard=np.zeros((self.size, self.size))  # 从新初始化棋盘
                        # print("玩家1走了一步")
                        next_observation = self.chessboard
                        reward = 0.0
                        done = False
                        info = {}
                        return next_observation, reward, done, info
            else:
                print("不能同时走两步","player:",player,"lastTimeStep",self.lastTimeStep)


        elif player == 2:
            if self.lastTimeStep == 1:
                self.lastTimeStep=2
                done = self.checkFoul(2, action)  # 在已落子格子上落子
                if done:
                    # print("玩家2犯规，游戏重启")
                    self.lastTimeStep = 2  # 游戏重启，玩家1先走
                    self.chessboard = np.zeros((self.size, self.size))  # 从新初始化棋盘
                    next_observation = self.chessboard
                    reward = -0.01
                    done = True
                    info = {}
                    return next_observation, reward, done, info
                else:
                    win = self.checkWin(player)
                    if win:
                        # print("玩家2赢，游戏重置")
                        self.chessboard = np.zeros((self.size, self.size))  # 从新初始化棋盘
                        next_observation = self.chessboard
                        reward = 1.0
                        done = True
                        info = {}
                        return next_observation, reward, done, info
                    else:
                        draw = self.checkDraw()
                        if draw:
                            self.chessboard = np.zeros((self.size, self.size))  # 从新初始化棋盘
                        # print("玩家2走了一步")
                        next_observation = self.chessboard
                        reward = 0.0
                        done = False
                        info = {}
                        return next_observation, reward, done, info
            else:
                print("不能同时走两步")

        else:
            print("输入玩家编号错误，玩家编号为1或者2")

    def checkFoul(self,player,action):#检查是否犯规
        row = action//self.size  #行索引
        Column = action%self.size  #列索引
        if self.chessboard[row][Column] != 0:
            return True #犯规，在已经有子的格子上落子
        else:
            self.chessboard[row][Column]=player
            return False

    #平局
    def checkDraw(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.chessboard[i][j] == 0:
                    return False
        return True

    #赢句
    def checkWin(self,winPlayer):
        #横轴
        for i in range(self.size):#行可取满
            for j in range(self.size-4):    #列不可取满
                resultRow = self.checkWinRow(i, j,winPlayer)

                if resultRow:#检测出五子连珠
                    # print("***************************************")
                    return resultRow
        #纵轴方向
        for i in range(self.size-4):#行不可取满
            for j in range(self.size):#列可取满
                resultCol = self.checkWinCol(i, j,winPlayer)
                if resultCol:#检测出五子连珠
                    return resultCol
        #右下方向
        for i in range(self.size-4):
            for j in range(self.size-4):
                resultRightLow = self.checkWinRightLow(i, j,winPlayer)
                if resultRightLow:#检测出五子连珠
                    return resultRightLow
        #右上方向
        for i in range(4,self.size):
            for j in range(self.size-4):
                resultRightHigh = self.checkWinRightHigh(i, j,winPlayer)
                if resultRightHigh:#检测出五子连珠
                    return resultRightHigh

        return False



    def checkWinRow(self,i,j,winPlayer):
        winValue=self.chessboard[i][j]
        if winValue != winPlayer:
            return False


        if self.isValue(i,j+1,winValue) and self.isValue(i,j+2,winValue) and self.isValue(i,j+3,winValue) and self.isValue(i,j+4,winValue) :
            #五子同值
            print("五个{0}值横轴方向方向连珠".format(winValue))
            return True

        else:
            return False

    def checkWinCol(self,i,j,winPlayer):
        winValue=self.chessboard[i][j]
        if winValue != winPlayer:
            return False

        if self.isValue(i+1,j,winValue) and self.isValue(i+2,j,winValue) and self.isValue(i+3,j,winValue) and self.isValue(i+4,j,winValue):
            #五子同值
            print("五个{0}值纵轴方向方向连珠".format(winValue))
            return True

        else:
            return False

    def checkWinRightLow(self,i,j,winPlayer):
        winValue=self.chessboard[i][j]
        if winValue != winPlayer:
            return False

        if self.isValue(i+1,j+1,winValue) and self.isValue(i+2, j+2, winValue) and self.isValue(i+3,j+3,winValue) and self.isValue(i+4,j+4,winValue) :
            #五子同值
            print("五个{0}值右下方向连珠".format(winValue))
            return True

        else:
            return False

    def checkWinRightHigh(self,i,j,winPlayer):
        winValue=self.chessboard[i][j]
        if winValue != winPlayer:
            return False

        if self.isValue(i-1,j+1,winValue) and self.isValue(i-2, j+2, winValue) and self.isValue(i-3,j+3,winValue) and self.isValue(i-4,j+4,winValue) :
            #五子同值
            print("五个{0}值右上方向连珠".format(winValue))
            return True

        else:
            return False

    def isValue(self,i,j,value):
        if self.chessboard[i][j] == value:
            return True
        else:
            return False

    def showChessboard(self):
        print(self.chessboard)

    def reset(self):
        self.chessboard=np.zeros((self.size, self.size))
        next_observation=self.chessboard
        self.lastTimeStep = 2
        return next_observation

    def sample(self):
        return np.random.randint(0,self.size*self.size)

if __name__ == '__main__':
    board=Gomoku(6)
    board.showChessboard()
    while True:
        actiony = input("玩家1请输入行：")
        actionx = input("玩家1请输入列：")
        action=int(actiony)*6+int(actionx)
        action = int(action)
        observation, reward, done, info=board.step(1,action)
        # print(type(observation))
        # print(observation.shape)
        # observation=observation.reshape(36)
        # print(observation.shape)
        # print(observation)
        # exit()
        board.showChessboard()
        actiony = input("玩家2请输入行：")
        actionx = input("玩家2请输入列：")
        action = int(actiony) * 6 + int(actionx)
        action = int(action)
        board.step(2, action)
        board.showChessboard()

