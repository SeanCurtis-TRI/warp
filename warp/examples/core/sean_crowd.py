import math

import numpy as np
import warp as wp
import warp.render

domain_width = wp.constant(20.0)
domain_height = wp.constant(20.0)
radius = wp.constant(0.2)

@wp.func
def compute_single_wall_force(p: wp.vec2, wall_pos: wp.vec2, wall_normal: wp.vec2):
    """Computes the force on particle p due to a single half space wall.
    The wall is defined by a point on the wall (wall_pos) and a normal pointing
    outward from the wall (wall_normal)."""
    dist = wp.dot(p - wall_pos, wall_normal)

    # TODO: This doesn't handle the case where the agent is within the wall.
    obstacle_gain = 2.0
    effect_radius = 4.0 * radius
    return wall_normal * (obstacle_gain * wp.exp(radius - dist) / effect_radius)


@wp.func
def compute_wall_forces(p: wp.vec2):
    force = compute_single_wall_force(p, wp.vec2(-domain_width * 0.5, 0.0),
                                      wp.vec2(1.0, 0.0))
    force += compute_single_wall_force(p, wp.vec2(domain_width * 0.5, 0.0),
                                       wp.vec2(-1.0, 0.0))
    force += compute_single_wall_force(p, wp.vec2(0.0, -domain_height * 0.5),
                                       wp.vec2(0.0, 1.0))
    force += compute_single_wall_force(p, wp.vec2(0.0, domain_height * 0.5),
                                       wp.vec2(0.0, -1.0))
    return force

@wp.kernel
def compute_accel(p: wp.array(dtype=wp.vec2), a: wp.array(dtype=wp.vec2)):
    id = wp.tid()
    p0 = p[id]
    # We assume unit mass, so a = f / 1.0.
    a_val = compute_wall_forces(p0)
    # TODO: This should be distance-based agent-agent calculations
    a[id] = a_val


@wp.kernel
def integrate(p: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
              a: wp.array(dtype=wp.vec2), dt: float):
    id = wp.tid()
    p[id] += v[id] * dt
    v[id] += a[id] * dt


class Example:
    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents
        measure = min(domain_width, domain_height)
        init_positions = (np.random.rand(num_agents, 2) - 0.5) * measure
        self.positions = wp.array(init_positions, dtype=wp.vec2)

        distances = np.sqrt(np.sum(init_positions**2, axis=1))
        distances.shape = (-1, 1)
        dirs = init_positions / distances
        init_velocities = np.empty_like(init_positions)
        init_velocities[:, 0] = dirs[:, 1]
        init_velocities[:, 1] = -dirs[:, 0]
        self.velocities = wp.array(init_velocities, dtype=wp.vec2)
        self.accels = wp.zeros_like(self.positions)

        self.agent_radius = radius
        self.desired_speed = 1.0

        self.dt = 0.05

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
        half_width = domain_width * 0.5
        half_height = domain_height * 0.5
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)
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

        