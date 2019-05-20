import numpy as np
r=[[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,100],
   [-1,-1,-1,0,-1,-1],
   [-1,0,0,-1,0,-1],
   [0,-1,-1,0,-1,100],
   [-1,0,-1,-1,0,100]]

npr=np.array(r)
npq=np.zeros((6,6))

def getAcion(state):
    action = np.random.randint(0, 6)
    if npr[state][action] > -1:
        return action
    else:
        return getAcion(state)


def qLeaning(state):
    action = getAcion(state)
    npq[state][action]=npr[state][action]+0.8*readQ(action).max()
    if action == 5:
        return
    else:
        qLeaning(action)



def readQ(state):
    a=[]
    for i in range(6):
        if npr[state][i]>-1:
            a.append(npq[state][i])

    return np.array(a)

for i in range(10000):
    init_state = np.random.randint(0, 6)
    qLeaning(init_state)
    print(npq)