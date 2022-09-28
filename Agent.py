import random
import math
import torch
from models import *

MULT = 1
DIP = 1
TARGET_DIST = 0.15
HYP = math.sqrt(TARGET_DIST**2 + TARGET_DIST**2)

USE = "EQU" # SIM|GNN|EQU
CASE = "HEX" # HEX|SQU

class Neighbor:

    def __init__(self, id, agent, other):
        self.id = id
        self.pose = other.pose
        self.dist = math.sqrt(pow(agent.pose[0]-other.pose[0],2)+pow(agent.pose[1]-other.pose[1], 2))
        angle = math.pi-math.atan2(agent.pose[1]-other.pose[1], agent.pose[0]-other.pose[0])
        self.azi = -(angle % (math.pi*2))

        if self.azi > math.pi:
            self.azi = math.pi - self.azi
        self.swarm = other.swarm
        self.message = 0

class Agent:

    def __init__(self, x=None, y=None, heading=None, swarm=None):
        if x is None:
            x = random.random()
        if y is None:
            y = random.random()
        if heading is None:
            heading = random.random()*2*math.pi - math.pi
        if swarm is None or CASE=="HEX":
            swarm = 1

        self.pose = [x, y, heading]
        self.velocity = [0, 0, 0]
        self.acceleration = [0, 0, 0]
        self.force = [0, 0, 0]       # x-> longitudinal, y-> latitudinal, h-> angular velocity
        self.swarm = swarm
        self.neighbors = []
        self.senseRange = 0.75          # 1 == length of world
        self.trajectory = []
        self.neighborHistory = []
        self.accelHistory = []

        self.eps = -1
        self.tar = -1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.squ_mod = OGNEDGE(2, 100, 2, hidden=300, aggr='add').to(self.device)
        #self.squ_mod.load_state_dict(torch.load('./SQU.mod'))
        self.squ_mod.eval()
        self.hex_mod = OGN(2, 100, 2, dt=0.1, hidden=300, edge_index=400, aggr='add').to(self.device)
        #self.hex_mod.load_state_dict(torch.load('./HEX.mod'))
        self.hex_mod.eval()

    def step(self):
        # record
        self.trajectory.append((self.pose[0], self.pose[1], self.pose[2]))

        # run controller
        if CASE == "HEX":
            self.hexLattice()
        elif CASE == "SQU":
            self.squareLattice()

        self.centerPull()

    def squareLattice(self):
        nHis = []
        x = 0
        y = 0
        gnn = [[0, 0] for _ in range(len(self.neighbors) + 1)]
        e1 = []
        e2 = []
        a = []
        gnn[0] = [self.pose[0], self.pose[1]]
        for i in range(len(self.neighbors)):
            n=self.neighbors[i]
            if USE == "SIM":
                d = TARGET_DIST
                s = 2
                if n.swarm == self.swarm:
                    d = HYP
                    s = 1
                nHis.append((n.id-1, s))
                mag = lj(n.dist, target=d)
                azi = n.azi+math.pi
                x = x + (mag*math.cos(azi))
                y = y + (mag*math.sin(azi))
            elif USE == "EQU":
                if n.swarm == self.swarm:
                    mag = (6.8e-7/pow(n.dist, 9.75) - (1.9e-5/pow(n.dist,7.75)))
                else:
                    mag = (1.58e-9/pow(n.dist,10.67)-(4e-6/pow(n.dist,6.79)))
                azi = n.azi+math.pi
                x += (mag*math.cos(azi))
                y += (mag*math.sin(azi))
            elif USE == "GNN":
                gnn[i + 1] = [n.dist * math.cos(n.azi) + self.pose[0], n.dist * math.sin(n.azi) + self.pose[1]]
                e1.extend([0, i + 1])
                e2.extend([i + 1, 0])
                a.extend([1, 1] if self.swarm == n.swarm else [2, 2])

        if USE=="SIM" or USE=="EQU":
            if len(self.neighbors) > 0:
                x = x / len(self.neighbors)
                y = y / len(self.neighbors)
        elif USE=="GNN":
            gnnData = torch.tensor(gnn, dtype=torch.float)
            e = torch.tensor([e1, e2], dtype=torch.long)
            a = torch.tensor(a, dtype=torch.float)
            pred = self.squ_mod(gnnData, e, a)
            x = pred[0][0].item()
            y = pred[0][1].item()

        self.neighborHistory.append(nHis)
        self.force[0] = x
        self.force[1] = y

    def hexLattice(self):
        nHis = []
        x = 0
        y = 0
        gnn = [[0, 0] for _ in range(len(self.neighbors)+1)]
        e1 = []
        e2 = []
        gnn[0] = [self.pose[0], self.pose[1]]
        for i in range(len(self.neighbors)):
            n = self.neighbors[i]
            if USE == "SIM":
                nHis.append(n.id)
                mag = lj(n.dist)
                azi = n.azi+math.pi
                x = x + (mag*math.cos(azi))
                y = y + (mag*math.sin(azi))
            elif USE == "EQU":
                mag = (8e-9/pow(n.dist, 10.07)) - (9.8e-6/pow(n.dist, 6.54))
                azi = n.azi+math.pi
                x += (mag*math.cos(azi))
                y += (mag*math.sin(azi))
            elif USE == "GNN":
                gnn[i+1] = [n.dist*math.cos(n.azi)+self.pose[0], n.dist*math.sin(n.azi)+self.pose[1]]
                e1.extend([0, i+1])
                e2.extend([i+1, 0])

        if USE=="SIM" or USE=="EQU":
            if len(self.neighbors) > 0:
                x = x / len(self.neighbors)
                y = y / len(self.neighbors)
        elif USE=="GNN":
            gnnData = torch.tensor(gnn, dtype=torch.float)
            e = torch.tensor([e1, e2], dtype=torch.long)
            pred = self.hex_mod(gnnData, e)
            x = pred[0][0].item()
            y = pred[0][1].item()

        self.neighborHistory.append(nHis)
        print(x, "::", y)
        self.force[0] = x
        self.force[1] = y

    def centerPull(self):
        self.force[0] += 1 * (0.5-self.pose[0])
        self.force[1] += 1 * (0.5-self.pose[1])

def lj(dist, target=None):
    if target:
        t = target
    else:
        t = TARGET_DIST
    epsilon = DIP * MULT
    sigma = (t * MULT) / (2 ** (1 / 6))
    mag = (epsilon * 4) * (pow((sigma / dist), 12) - pow((sigma / dist), 6))
    return min(mag, 10)#MULT)