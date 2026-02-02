import math

import numpy as np
import warp as wp
import warp.render

@wp.kernel
def compute_vel(p: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2)):
    id = wp.tid()
    p0 = p[id]
    # For now, circle the origin (explicit integration should cause badness).
    p0_hat = wp.normalize(p0)
    speed = 1.3
    v[id] = wp.vec2(-p0_hat.y, p0_hat.x) * speed


@wp.kernel
def integrate(p: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2), dt: float):
    id = wp.tid()
    p[id] += v[id] * dt


class Example:
    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents

        init_positions = np.random.rand(num_agents, 2) * 10.0 - 5
        self.positions = wp.array(init_positions, dtype=wp.vec2)

        self.velocities = wp.zeros_like(self.positions)

        self.agent_radius = 0.2
        self.desired_speed = 1.0

        self.dt = 0.1

    def step(self):
        with wp.ScopedTimer("step"):
            wp.launch(compute_vel, dim=self.num_agents,
                    inputs=[self.positions, self.velocities])
            wp.launch(integrate, dim=self.num_agents,
                    inputs=[self.positions, self.velocities, self.dt])

    def step_and_render_frame(self, frame_num=None, agents=None):
        self.step()
        
        # Update agent patches
        with wp.ScopedTimer("render"):
            if agents:
                positions = self.positions.numpy()
                for i, agent in enumerate(agents):
                    pos = positions[i]
                    agent.center = (pos[0], pos[1])

        return agents


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None,
                         help="Override the default Warp device.")
    parser.add_argument('--num_frames', type=int, default=800,
                        help="Total number of frames.")
    parser.add_argument('--num_agents', type=int, default=30,
                        help="Number of agents in the crowd.")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        example = Example(args.num_agents)
        import matplotlib.patches as patches
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        agents = []

        fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal') # Important for circles to appear round

        # Add circles as patches
        for pi in example.positions.numpy():
            c = np.random.rand(3,)
            circle = patches.Circle(pi, radius=example.agent_radius, color=c)
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

        