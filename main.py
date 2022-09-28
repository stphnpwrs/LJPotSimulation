from Visualizer import Visualizer
from World import World
from Agent import Agent
import pickle
from matplotlib import pyplot as plt

NUMAGENTS = 20
NUMEXPERIMENTS = 1
TIME = 25 # seconds
ANALYZE = False
VISUAL = True

def makeNeighborLink(d):
    traj = d['traj']
    nHis = d['nHis']
    acc = d['acc']
    for i in range(len(traj)): # for each agent                     i -> agent id
        for j in range(len(traj[i])): # for each timestep             j -> timestep
            if i == 0:
                #print(nHis[i][j])
                #print(traj[i][j])
                if acc[i][j][0] > 1 or acc[i][j][0] < -1:
                    print(acc[i][j])
            #break

for i in range(NUMEXPERIMENTS):
    if i%100== 0:
        print(str(i)+"/"+str(NUMEXPERIMENTS))
    filename = "./log"+str(i)+".p"

    if ANALYZE:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            makeNeighborLink(data)
    else:
        with open(filename, "wb") as f:
            world = World(f)
            for j in range(int(NUMAGENTS)):
                a = Agent(swarm=j%2)
                world.agents.append(a)

            if VISUAL:
                v = Visualizer(world)
                v.run()
            else:
                while world.stepCount < TIME*10:
                    world.step()
                world.closeWorld()
