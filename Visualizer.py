import pygame
import math

BL = (0, 0, 0)

class Visualizer:

    def __init__(self, world, logFile=None):
        pygame.init()
        self.width = 400
        self.height = 400
        self.display = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption('Visualizer')
        self.clock = pygame.time.Clock()
        self.done = False
        self.world = world
        self.logFile = logFile

    def run(self):

        while not self.done:
            # check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True

            # update world
            self.world.step()

            # draw agents
            self.display.fill((255, 255, 255))
            for agent in self.world.agents:
                x = int(agent.pose[0]*self.width)
                y = int((agent.pose[1])*self.height)
                h = agent.pose[2]
                radius = 10
                pygame.draw.circle(self.display, color(agent.swarm), (x,y), radius, 0)
                pygame.draw.circle(self.display, BL, (x,y), radius, 2)
                pygame.draw.line(self.display, BL, (x, y),
                                   (x + radius*math.cos(h),
                                    y + radius*math.sin(h)), 2)
                #if agent == self.world.agents[0]:
                #self.showNeighborNet(agent)
                #self.showForce(agent)
            pygame.display.update()

            # tick -> fps == simulation step
            self.clock.tick(self.world.hz)

        self.world.closeWorld()
        pygame.quit()

    def showNeighborNet(self, agent):
        a = (int(agent.pose[0]*self.width), int(agent.pose[1]*self.height))
        for n in agent.neighbors:
            b = (int(n.pose[0]*self.width), int(n.pose[1]*self.height))
            pygame.draw.line(self.display, (0,0,0), a, b, 2)

            x = agent.pose[0]
            y = agent.pose[1]
            h = agent.pose[2]

            newX = x + n.dist*math.cos(n.azi)
            newY = y + n.dist*math.sin(n.azi)

            d = (int(newX*self.width), int(newY*self.height))

            pygame.draw.line(self.display, (0,0,255), a, d, 2)

    def showForce(self, agent):
        a = (int(agent.pose[0] * self.width), int(agent.pose[1] * self.height))

        fx = agent.force[0]
        fy = agent.force[1]

        x = agent.pose[0] + fx
        y = agent.pose[1] + fy

        b = (int(x*self.width), int(y * self.height))

        pygame.draw.line(self.display, (255, 0, 0), a, b, 2)

# Swarm Colors
def color(swarm):
    if swarm == 0:
        return (255, 0, 0)
    if swarm == 1:
        return (0, 255, 0)
    if swarm == 2:
        return (0, 0, 255)
    if swarm == 3:
        return (255, 255, 0)
    if swarm == 4:
        return (255, 0, 255)
    if swarm == 5:
        return (0, 255, 255)
    if swarm == 6:
        return (0, 0, 0)
    if swarm == 7:
        return (255, 255, 255)
