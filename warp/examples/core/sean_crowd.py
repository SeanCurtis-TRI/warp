import math

import numpy as np
import warp as wp
import warp.render


class Example:
    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents

        self.init_positions = np.random.rand(num_agents, 2) * 10.0 - 5
        self.positions = np.copy(self.init_positions)
        # self.positions = wp.array(self.init_positions, dtype=wp.vec2)
        # self.velocities = wp.array(
        #     [[0.0, 0.0] for _ in range(num_agents)], dtype=wp.vec2
        # )

        self.agent_radius = 0.2
        self.desired_speed = 1.0

        self.dt = 0.1

    def step(self):
        shape = self.positions.shape
        speed = 1.3  # m/s
        velocities = (np.random.rand(shape[0], shape[1]) - 0.5) * speed
        self.positions += velocities * self.dt

    def step_and_render_frame(self, frame_num=None, agents=None):
        self.step()
        
        # Update agent patches
        if agents:
            for i, agent in enumerate(agents):
                pos = self.positions[i]
                agent.center = (pos[0], pos[1])

        return agents


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None,
                         help="Override the default Warp device.")
    parser.add_argument('--num_frames', type=int, default=800,
                        help="Total number of frames.")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        example = Example(30)
        import matplotlib.patches as patches
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        agents = []

        fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal') # Important for circles to appear round

        # Add circles as patches
        for pi in example.init_positions:
            circle = patches.Circle(pi, radius=example.agent_radius, color='b')
            ax.add_patch(circle)
            agents.append(circle)

        seq = anim.FuncAnimation(
            fig,
            example.step_and_render_frame,
            fargs=(agents,),
            frames=args.num_frames,
            blit=True,
            interval=8,
        )

        plt.show()

        