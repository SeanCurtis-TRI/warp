import math

import numpy as np
import warp as wp
import warp.render

@wp.kernel
def compute_accel(p: wp.array(dtype=wp.vec2), a: wp.array(dtype=wp.vec2)):
    id = wp.tid()
    p0 = p[id]
    p_AO = wp.normalize(-p0)
    # TODO: This should be distance-based agent-agent calculations
    # TODO: Include agent-border calculations
    # Currently, we accelerate towards the origin.
    a[id] = p_AO * 0.5


@wp.kernel
def integrate(p: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
              a: wp.array(dtype=wp.vec2), dt: float):
    id = wp.tid()
    p[id] += v[id] * dt
    v[id] += a[id] * dt


class Example:
    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents

        init_positions = np.random.rand(num_agents, 2) * 10.0 - 5
        self.positions = wp.array(init_positions, dtype=wp.vec2)

        distances = np.sqrt(np.sum(init_positions**2, axis=1))
        distances.shape = (-1, 1)
        dirs = init_positions / distances
        init_velocities = np.empty_like(init_positions)
        init_velocities[:, 0] = dirs[:, 1]
        init_velocities[:, 1] = -dirs[:, 0]
        self.velocities = wp.array(init_velocities, dtype=wp.vec2)
        self.accels = wp.zeros_like(self.positions)

        self.agent_radius = 0.2
        self.desired_speed = 1.0

        self.dt = 0.1

    def step(self):
        with wp.ScopedTimer("step"):
            wp.launch(compute_accel, dim=self.num_agents,
                    inputs=[self.positions, self.accels])
            wp.launch(integrate, dim=self.num_agents,
                    inputs=[self.positions, self.velocities, self.accels,self.dt])

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

        