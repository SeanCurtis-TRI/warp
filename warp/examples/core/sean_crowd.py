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

# Density field
field_width = wp.constant(128)
field_height = wp.constant(128)
cell_area = wp.constant((domain_width * domain_height) / (field_width * field_height))
rho_kernel_size = wp.constant(4.0)  # in meters (the measure of the compact support of the density kernel.)


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


@wp.kernel
def update_box_density(p: wp.array(dtype=wp.vec2), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    fw = float(field_width)
    fh = float(field_height)
    rho_support = rho_kernel_size
    kSize = int(rho_support * float(field_width) / domain_width + 0.5)
    kDelta = kSize // 2
    cell_population = 1.0 / float(rho_support * rho_support)
    for a in range(len(p)):
        pos = p[a]
        # Map position to field coordinates.
        x = int(((pos.x + (domain_width * 0.5)) / domain_width) * fw + 0.5)
        y = int(((pos.y + (domain_height * 0.5)) / domain_height) * fh + 0.5)
        # Splat each agent into a kxk block.
        if x >= i - kDelta and x <= i + kDelta and y >= j - kDelta and y <= j + kDelta:
            rho[j, i] += cell_population


@wp.kernel
def update_circle_density(p: wp.array(dtype=wp.vec2), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    cx = (float(i) + 0.5) * domain_width / float(field_width) - (domain_width * 0.5)
    cy = (float(j) + 0.5) * domain_height / float(field_height) - (domain_height * 0.5)
    c = wp.vec2(cx, cy)
    rho_support = rho_kernel_size / 2.0
    support_sq = rho_support * rho_support
    cell_population = 1.0 / (np.pi * support_sq)
    for a in range(len(p)):
        pos = p[a]
        dist_sq = wp.length_sq(pos - c)
        if dist_sq <= support_sq:
            rho[j, i] += cell_population


@wp.kernel
def update_first_order_density(p: wp.array(dtype=wp.vec2), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    cx = (float(i) + 0.5) * domain_width / float(field_width) - (domain_width * 0.5)
    cy = (float(j) + 0.5) * domain_height / float(field_height) - (domain_height * 0.5)
    c = wp.vec2(cx, cy)
    rho_support = rho_kernel_size / 2.0
    support_sq = rho_support * rho_support
    cone_vol = np.pi * support_sq * rho_support / 3.0
    norm = 1.0 / cone_vol
    for a in range(len(p)):
        pos = p[a]
        dist_sq = wp.length_sq(pos - c)
        if dist_sq <= support_sq:
            dist = rho_support -wp.sqrt(dist_sq)
            rho[j, i] += dist * norm


@wp.kernel
def update_gauss_density(p: wp.array(dtype=wp.vec2), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    cx = (float(i) + 0.5) * domain_width / float(field_width) - (domain_width * 0.5)
    cy = (float(j) + 0.5) * domain_height / float(field_height) - (domain_height * 0.5)
    c = wp.vec2(cx, cy)
    rho_support = rho_kernel_size / 2.0
    support_sq = rho_support * rho_support
    sigma = rho_support / 3.0
    # A *normalized* gaussian function in 2D, integrated over all space is 1.
    # Integrated over a circle of radius 3σ is ~0.9889. We'll include the
    # scale factor and the normalization factor together.
    # A 2D gaussian function integrated on a circle of radius 3sigma has an
    # integral of 0.9889. So, we'll hard-code it here and normalize. We could
    # probably get away without doing it, but one more multiplication per
    # agent is not going to hurt.
    two_sigma_sq = 2.0 * sigma * sigma
    norm = 1.0 / (0.9889 * two_sigma_sq * np.pi)
    for a in range(len(p)):
        pos = p[a]
        dist_sq = wp.length_sq(pos - c)
        if dist_sq <= support_sq:
            val = wp.exp(-dist_sq / two_sigma_sq)
            rho[j, i] += val * norm


class Scenario:
    def __init__(self, positions, velocities, goals, colors):
        assert len(positions) == len(velocities) == len(goals) == len(colors)
        self.positions = positions
        self.velocities = velocities
        self.goals = goals
        self.colors = colors
        self.density_kernel = update_box_density

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
        theta = i * dtheta + 0.35
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

        self.density_kernel = scenario.density_kernel
        self.density = wp.zeros((field_width, field_height), dtype=float)

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
        return True

    def update_density(self):
        wp.launch(self.density_kernel, dim=(field_width, field_height),
                  inputs=[self.positions, self.density])

    def map_to_field(self, p):
        """Given a position in "world" space, map it to the density field.

        [-hw, -hhw]x[hw, hh] --> [0, 0]x[field_width, field_height]
        """
        p += np.array(((domain_width * 0.5, domain_height * 0.5),))
        p *= np.array(((field_width / domain_width, field_height / domain_height),))
        return p

    def step_and_render_frame(self, frame_num=None, agents=None, density_img=None):
        running = self.step()
        
        # Update agent patches
        with wp.ScopedTimer("render"):
            if agents:
                positions = self.map_to_field(self.positions.numpy())
                for i, agent in enumerate(agents):
                    pos = positions[i]
                    agent.center = (pos[0], pos[1])
        if density_img:
            with wp.ScopedTimer("density"):
                self.update_density()
                rho = self.density.numpy()
                density_img.set_array(rho)
                print(f"Total population = {(cell_area * rho.sum()):.2f}")

        if not running:
            global seq
            seq.event_source.stop()
        return agents + [density_img]


if __name__ == '__main__':
    import argparse

    kernels = {
        'box': update_box_density,
        'circle': update_circle_density,
        'first_order': update_first_order_density,
        'gauss': update_gauss_density,
    }

    scenarios = {'random': random_scenario,
                 'circle': circle_scenario,
                 'four_blocks': four_blocks_scenario,
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
    parser.add_argument('--density', type=str, choices=list(kernels.keys()),
                        default='first_order',
                        help=f"Choose a density-field kernel: {', '.join(kernels.keys())}")
    parser.add_argument('--density_size', type=float, default=4.0,
                        help="The width of the density kernel support in meters")
    args = parser.parse_args()
    rho_kernel_size = wp.constant(args.density_size)


    with wp.ScopedDevice(args.device):
        import matplotlib
        import matplotlib.patches as patches
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        scenario = scenarios[args.scenario](args.num_agents)
        scenario.density_kernel = kernels[args.density]
        sim = Simulation(scenario)

        agents = []

        fig, ax = plt.subplots()

        img = plt.imshow(
            sim.density.numpy(),
            origin="lower",
            animated=True,
            interpolation="antialiased",
        )
        img.set_norm(matplotlib.colors.Normalize(0.0, 3))
        plt.colorbar(img, label='ρ (people/m²)')

        # Change the axis ticks to be simulation world coordinates.
        x_ticks = [0, field_width * 0.25, field_width * 0.5, field_width * 0.75, field_width]
        x_labels = [-domain_width * 0.5, -domain_width * 0.25, 0.0, domain_width * 0.25, domain_width * 0.5]
        ax.set_xticks(x_ticks, x_labels)
        y_ticks = [0, field_height * 0.25, field_height * 0.5, field_height * 0.75, field_height]
        y_labels = [-domain_height * 0.5, -domain_height * 0.25, 0.0, domain_height * 0.25, domain_height * 0.5]
        ax.set_yticks(y_ticks, y_labels)
        ax.set_aspect('equal') # Important for circles to appear round

        # Add circles as patches.
        # We're scaling the radius to the field dimensions. We're assuming that
        # the field and simulation domain have the same aspect ratio.
        r = sim.agent_radius * field_width / domain_width
        for i, pi in enumerate(sim.positions.numpy()):
            circle = patches.Circle(pi, radius=r, color=scenario.colors[i, :])
            ax.add_patch(circle)
            agents.append(circle)

        seq = anim.FuncAnimation(
            fig,
            sim.step_and_render_frame,
            fargs=(agents, img),
            frames=args.num_frames,
            blit=True,
            interval=1,
        )

        plt.show()

        