import numpy as np
import warp as wp
import warp.render

domain_width = wp.constant(20.0)
domain_height = wp.constant(20.0)

# Common parameters
radius = wp.constant(0.2)  # m
neighbor_distance = wp.constant(5.0)  # m
pref_speed = wp.constant(1.3)   # m/s
max_speed = wp.constant(2.0)  # m/s

# Helbing parameters
mass = wp.constant(80.0)
# The scales in Menge are 10X what they are here.
agent_scale = wp.constant(200.0)
obstacle_scale = wp.constant(400.0)
reaction_time = wp.constant(0.5)
# Note: I've got 0.015 in Menge; this doesn't work well here. Not sure; don't care.
force_distance = wp.constant(radius * 10)


@wp.func
def compute_single_wall_force(p: wp.vec2, wall_pos: wp.vec2, wall_normal: wp.vec2):
    """Computes the force on particle p due to a single half space wall.
    The wall is defined by a point on the wall (wall_pos) and a normal pointing
    outward from the wall (wall_normal)."""
    dist = wp.dot(p - wall_pos, wall_normal)

    # TODO: This doesn't handle the case where the agent is within the wall.
    # The force always points *out* of the half space, but the deeper they get
    # the stronger the force becomes. It would be better to compute just the
    # force necessary to accelerate it out of the wall.
    return wall_normal * (obstacle_scale * wp.exp((radius - dist) / force_distance))


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


@wp.func
def compute_single_agent_force(p: wp.vec2, q: wp.vec2):
    """Computes the repulsive force on agent at position p due to another agent
    at position q."""
    r_QP = p - q
    dist_QP = wp.norm_l2(r_QP)
    if dist_QP > neighbor_distance or dist_QP < 1e-5:
        return wp.vec2(0.0, 0.0)
    mag = agent_scale * wp.exp(((2.0 * radius) - dist_QP) / force_distance)
    mag = min(mag, 1e5)
    return r_QP * (mag / dist_QP)


@wp.func
def compute_agent_forces(id: int, p: wp.array(dtype=wp.vec2)):
    force = wp.vec2(0.0, 0.0)
    for i in range(len(p)):
        if i == id:
            continue
        force += compute_single_agent_force(p[id], p[i])

    return force


@wp.func
def compute_desired_velocity(p: wp.vec2, g: wp.vec2, dt: float):
    """Computes the desired velocity vector for an agent at position p towards
    goal g."""
    to_goal = g - p
    dist_to_goal = wp.norm_l2(to_goal)
    if dist_to_goal < 1e-5:
        return wp.vec2(0.0, 0.0)
    speed = min(pref_speed, dist_to_goal / dt)
    desired_velocity = to_goal * (speed / dist_to_goal)
    return desired_velocity


@wp.func
def compute_driving_force(id: int, p: wp.array(dtype=wp.vec2),
                          v: wp.array(dtype=wp.vec2),
                          g: wp.array(dtype=wp.vec2), dt: float):
    """Computes the driving force for all agents towards their goals."""
    v_pref = compute_desired_velocity(p[id], g[id], dt)
    force = mass * (v_pref - v[id]) / reaction_time  # Assuming unit mass here.
    return force


@wp.kernel
def compute_accel(p: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
                  a: wp.array(dtype=wp.vec2), g: wp.array(dtype=wp.vec2),
                  dt: float):
    id = wp.tid()
    p0 = p[id]
    # We assume unit mass, so a = f / 1.0.
    f = compute_wall_forces(p0)
    f += compute_agent_forces(id, p)
    f += compute_driving_force(id, p, v, g, dt)
    a[id] = f / mass


@wp.kernel
def integrate(p: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
              a: wp.array(dtype=wp.vec2), dt: float):
    id = wp.tid()
    p[id] += v[id] * dt
    v[id] += a[id] * dt
    # Clamp speed; it doesn't get to accelerate indefinitely.
    new_speed = wp.norm_l2(v[id])
    if new_speed > max_speed:
        v[id] = v[id] * (max_speed / new_speed)


class Scenario:
    def __init__(self, positions, velocities, goals, colors):
        assert len(positions) == len(velocities) == len(goals) == len(colors)
        self.positions = positions
        self.velocities = velocities
        self.goals = goals
        self.colors = colors

# Functions for creating the initial conditions of scenarios. Each returns
# initial positions, velocities, goals, and per-agent colors.

def random_scenario(num_agents: int):
    # Initial positions randomly distributed through the domain.
    measure = min(domain_width, domain_height)
    positions = (np.random.rand(num_agents, 2) - 0.5) * measure
    velocities = np.zeros_like(positions)
    goals = -positions

    # Random colors.
    colors = np.random.rand(num_agents, 3)

    return Scenario(positions, velocities, goals, colors)


def circle_scenario(num_agents: int):
    R = min(domain_width, domain_height) * 0.5  - force_distance
    max_agents = int(2 * np.pi * R / (2.5 * radius))
    if num_agents > max_agents:
        print(f"Warning: Reducing number of agents from {num_agents} to "
              f"{max_agents} to fit in circle scenario.")
        num_agents = max_agents

    positions = np.empty((num_agents, 2), dtype=np.float32)
    velocities = np.zeros_like(positions)
    goals = np.empty_like(positions)
    colors = np.empty((num_agents, 3), dtype=np.float32)

    dtheta = 2.0 * np.pi / num_agents
    for i in range(num_agents):
        theta = i * dtheta
        c = np.cos(theta)
        s = np.sin(theta)
        positions[i, 0] = R * c
        positions[i, 1] = R * s
        goals[i, :] = -positions[i, :]
        colors[i, :] = np.array([c * 0.5 + 0.5, s * 0.5 + 0.5, 0.5])
    return Scenario(positions, velocities, goals, colors)


def four_blocks_scenario(num_agents: int):
    print(f"Warning: Four blocks ignores the number of agents argument.")

    num_agents = 100 # 4 blocks of 25 agents each.
    positions = np.empty((num_agents, 2), dtype=np.float32)
    velocities = np.empty_like(positions)
    goals = np.empty_like(positions)
    colors = np.empty((num_agents, 3), dtype=np.float32)

    # Block measure is 3/4 of a quadrant.
    w = domain_width * 0.5 * 0.75
    dx = w / 4.0
    p0 = domain_width * 0.5 * 0.125
    quadrants = (
        (p0, dx, p0, dx, (1, 0, 0)),  # Top right
        (-p0, -dx, p0, dx, (0, 1, 0)),  # Top left
        (-p0, -dx, -p0, -dx, (0, 0, 1)),  # Bottom left
        (p0, dx, -p0, -dx, (1, 0, 1)),  # Bottom right
    )
    i = 0
    for x0, dx, y0, dy, c in quadrants:
         for ix in range(5):
            for iy in range(5):
                positions[i, 0] = x0 + ix * dx
                positions[i, 1] = y0 + iy * dy
                goals[i, :] = -positions[i, :]
                velocities[i, :] = 0.0
                colors[i, :] = c
                i += 1
    return Scenario(positions, velocities, goals, colors)


class Simulation:
    def __init__(self, scenario):
        self.num_agents = len(scenario.positions)
        self.positions = wp.array(scenario.positions, dtype=wp.vec2)
        self.velocities = wp.array(scenario.velocities, dtype=wp.vec2)
        self.accels = wp.zeros_like(self.positions)

        self.goals = wp.array(scenario.goals, dtype=wp.vec2)

        self.agent_radius = radius

        self.sub_steps = 25
        self.dt = 0.001 * self.sub_steps  # Effectively dt = 0.001

    def step(self):
        with wp.ScopedTimer("step"):
            sub_dt = self.dt / self.sub_steps
            for i in range(self.sub_steps):
                wp.launch(compute_accel, dim=self.num_agents,
                        inputs=[self.positions, self.velocities, self.accels,
                                self.goals, sub_dt]
                )
                wp.launch(integrate, dim=self.num_agents,
                        inputs=[self.positions, self.velocities, self.accels,
                                sub_dt]
                )
            v = self.velocities.numpy()
            if (np.abs(v) < 6e-2).all():
                print("All agents stopped moving!")
                return False
            else:
                print("Maximum speed componenent:", np.max(np.abs(v)))
        return True

    def step_and_render_frame(self, frame_num=None, agents=None):
        running = self.step()
        
        # Update agent patches
        with wp.ScopedTimer("render"):
            if agents:
                positions = self.positions.numpy()
                for i, agent in enumerate(agents):
                    pos = positions[i]
                    agent.center = (pos[0], pos[1])

        if not running:
            global seq
            seq.event_source.stop()
        return agents


if __name__ == '__main__':
    import argparse

    scenarios = {'random': random_scenario,
                 'circle': circle_scenario,
                 'four_blocks': four_blocks_scenario
                 }

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None,
                         help="Override the default Warp device.")
    parser.add_argument('--num_frames', type=int, default=800,
                        help="Total number of frames.")
    parser.add_argument('--num_agents', type=int, default=30,
                        help="Number of agents in the crowd.")
    parser.add_argument('--scenario', type=str, choices=list(scenarios.keys()),
                        default='circle',
                        help=f"Choose a scenario: {', '.join(scenarios.keys())}")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        scenario = scenarios[args.scenario](args.num_agents)
        Simulation = Simulation(scenario)
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
        for i, pi in enumerate(Simulation.positions.numpy()):
            circle = patches.Circle(pi, radius=Simulation.agent_radius,
                                    color=scenario.colors[i, :])
            ax.add_patch(circle)
            agents.append(circle)

        seq = anim.FuncAnimation(
            fig,
            Simulation.step_and_render_frame,
            fargs=(agents,),
            frames=args.num_frames,
            blit=True,
            interval=1,
        )

        plt.show()

        