import math
from Agent import Neighbor
import pickle

class World:

    def __init__(self, logfile, hz=10, verbose=False):
        self.agents = []
        self.hz = hz
        self.stepCount = 0
        self.maxAccel = .1 #.1
        self.maxVel = 0.025 #0.025
        self.maxAngAcc = (math.pi*2)/self.hz
        self.maxAngVel = (math.pi*2)/self.hz
        self.logfile = logfile
        self.verbose = verbose

    def step(self):
        for agent in self.agents:
            self.logStep()                                  # log the step for debugging
            self.updateNeighbors(agent, self.agents)        # update list of neighbors
            agent.step()                                    # updates Agent's control and force
            a = [agent.force[0],
                 agent.force[1],
                 agent.force[2]]                            # deep copy
            agent.accelHistory.append(a)                    # record acceleration of agent per step
            self.dynamics(agent)                      # applies Agent's force to agent
            self.applyPacman(agent)                         # applies boundaries to agent


        self.stepCount = self.stepCount+1

    def logStep(self):
        if self.verbose:  # log step if wanted
            print('=====================')
            print('Step: ' + str(self.stepCount))
            print('=====================')

    def applyPacman(self, agent):
        # pacman world
        agent.pose[0] = agent.pose[0] % 1
        agent.pose[1] = agent.pose[1] % 1
        agent.pose[2] = agent.pose[2] % (2 * math.pi)

    def updateNeighbors(self, agent, others):
        agent.neighbors = []
        c = 0
        for other in others:
            c = c + 1
            if agent == other:
                continue
            n = Neighbor(c, agent, other)
            if n.dist <= agent.senseRange:
                agent.neighbors.append(n)

    def dynamics(self, agent):

        agent.acceleration = limitVec(agent.force, self.maxAccel, self.maxAngAcc)
        agent.velocity[0] = agent.velocity[0] +(1.0/self.hz) * agent.acceleration[0]
        agent.velocity[1] = agent.velocity[1] +(1.0/self.hz) * agent.acceleration[1]
        agent.velocity[2] = agent.velocity[2] +(1.0/self.hz) * agent.acceleration[2]

        agent.velocity = limitVec(agent.velocity, self.maxVel, self.maxAngVel)
        agent.pose[0] = agent.pose[0] + (1.0/self.hz) * agent.velocity[0]
        agent.pose[1] = agent.pose[1] + (1.0/self.hz) * agent.velocity[1]
        agent.pose[2] = math.atan2(agent.velocity[1], agent.velocity[0])

    def closeWorld(self):

        traj = []
        neigh = []
        acc = []
        for a in self.agents:
            traj.append(a.trajectory)
            neigh.append(a.neighborHistory)
            acc.append(a.accelHistory)

        pickle.dump({'traj':traj, 'nHis':neigh, 'acc':acc}, self.logfile)

def limitVec(vec, val, valAng):

    a = math.atan2(vec[1], vec[0])
    m = math.sqrt(pow(vec[0], 2) + pow(vec[1], 2))
    m = max(min(m, val), -val)

    return [m * math.cos(a), m * math.sin(a), max(min(vec[2], valAng), -valAng)]