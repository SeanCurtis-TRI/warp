import sys

import numpy as np
import warp as wp
import warp.render

domain_size = wp.constant(20.0)

# Common parameters
radius = wp.constant(0.2)  # m
neighbor_distance = wp.constant(5.0)  # m
pref_speed = wp.constant(1.3)   # m/s
max_speed = wp.constant(2.0)  # m/s

# Helbing parameters
mass = wp.constant(80.0)
agent_scale = wp.constant(2000.0)
obstacle_scale = wp.constant(4000.0)
reaction_time = wp.constant(0.5)
# In Menge, this is hard-coded as 0.015, which is 3/4 of default radius.
# However, 0.5 radius seems to get me tighter crowds (with a 1e-3 time step,
# that seems alright).
force_distance = wp.constant(radius * 0.5)

# Density field
field_resolution = wp.constant(128)
cell_area = wp.constant((domain_size * domain_size) / (field_resolution * field_resolution))
rho_kernel_size = wp.constant(4.0)  # in meters (the measure of the compact support of the density kernel.)

# Some weird race condition can cause the step to evaluate without the sequence
# actually being created. So, we'll make sure the global exists and test for
# its definition below.
seq = None

def configure_constants(**kwargs):
    global domain_size
    global radius, neighbor_distance, pref_speed, max_speed
    global mass, agent_scale, obstacle_scale, reaction_time, force_distance
    global field_resolution, cell_area, rho_kernel_size

    for key, value in kwargs.items():
        if key == 'domain_size':
            domain_size = wp.constant(value)
        elif key == 'radius':
            radius = wp.constant(value)
        elif key == 'neighbor_distance':
            neighbor_distance = wp.constant(value)
        elif key == 'pref_speed':
            pref_speed = wp.constant(value)
        elif key == 'max_speed':
            max_speed = wp.constant(value)
        elif key == 'mass':
            mass = wp.constant(value)
        elif key == 'agent_scale':
            agent_scale = wp.constant(value)
        elif key == 'obstacle_scale':
            obstacle_scale = wp.constant(value)
        elif key == 'reaction_time':
            reaction_time = wp.constant(value)
        elif key == 'force_distance':
            force_distance = wp.constant(value)
        elif key == 'field_resolution':
            field_resolution = wp.constant(value)
        elif key == 'rho_kernel_size':
            rho_kernel_size = wp.constant(value)

    cell_area = wp.constant((domain_size * domain_size) / (field_resolution * field_resolution))

@wp.func
def compute_single_wall_force(p: wp.vec3, wall_pos: wp.vec3, wall_normal: wp.vec3):
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
def compute_wall_forces(p: wp.vec3):
    force = compute_single_wall_force(p, wp.vec3(-domain_size * 0.5, 0.0, 0.0),
                                      wp.vec3(1.0, 0.0, 0.0))
    force += compute_single_wall_force(p, wp.vec3(domain_size * 0.5, 0.0, 0.0),
                                       wp.vec3(-1.0, 0.0, 0.0))
    force += compute_single_wall_force(p, wp.vec3(0.0, -domain_size * 0.5, 0.0),
                                       wp.vec3(0.0, 1.0, 0.0))
    force += compute_single_wall_force(p, wp.vec3(0.0, domain_size * 0.5, 0.0),
                                       wp.vec3(0.0, -1.0, 0.0))
    return force


@wp.func
def compute_single_agent_force(p: wp.vec3, q: wp.vec3):
    """Computes the repulsive force on agent at position p due to another agent
    at position q."""
    r_QP = p - q
    dist_QP = wp.norm_l2(r_QP)
    if dist_QP > neighbor_distance or dist_QP < 1e-5:
        return wp.vec3(0.0, 0.0, 0.0)
    mag = agent_scale * wp.exp(((2.0 * radius) - dist_QP) / force_distance)
    mag = min(mag, 1e5)
    return r_QP * (mag / dist_QP)


@wp.func
def compute_agent_forces_grid(gid: int, p: wp.array(dtype=wp.vec3), grid: wp.uint64):
    force = wp.vec3(0.0, 0.0, 0.0)
    neighbors = wp.hash_grid_query(grid, p[gid], neighbor_distance)
    # for i in range(len(p)):
    for index in neighbors:
        if index != gid:
            force += compute_single_agent_force(p[gid], p[index])

    return force


@wp.func
def compute_agent_forces(id: int, p: wp.array(dtype=wp.vec3)):
    force = wp.vec3(0.0, 0.0, 0.0)
    for i in range(len(p)):
        if i != id:
            force += compute_single_agent_force(p[id], p[i])
    return force


@wp.func
def compute_desired_velocity(p: wp.vec3, g: wp.vec3, dt: float):
    """Computes the desired velocity vector for an agent at position p towards
    goal g."""
    to_goal = g - p
    dist_to_goal = wp.norm_l2(to_goal)
    if dist_to_goal < 1e-5:
        return wp.vec3(0.0, 0.0, 0.0)
    speed = min(pref_speed, dist_to_goal / dt)
    desired_velocity = to_goal * (speed / dist_to_goal)
    return desired_velocity


@wp.func
def compute_driving_force(id: int, p: wp.array(dtype=wp.vec3),
                          v: wp.array(dtype=wp.vec3),
                          g: wp.array(dtype=wp.vec3), dt: float):
    """Computes the driving force for all agents towards their goals."""
    v_pref = compute_desired_velocity(p[id], g[id], dt)
    force = mass * (v_pref - v[id]) / reaction_time  # Assuming unit mass here.
    return force


@wp.kernel
def compute_accel_grid(p: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                  a: wp.array(dtype=wp.vec3), g: wp.array(dtype=wp.vec3),
                  dt: float, grid: wp.uint64):
    tid = wp.tid()
    id = wp.hash_grid_point_id(grid, tid)
    p0 = p[id]
    f = compute_wall_forces(p0)
    f += compute_agent_forces_grid(id, p, grid)
    f += compute_driving_force(id, p, v, g, dt)
    a[id] = f / mass


@wp.kernel
def compute_accel(p: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                  a: wp.array(dtype=wp.vec3), g: wp.array(dtype=wp.vec3),
                  dt: float):
    id = wp.tid()
    p0 = p[id]
    # We assume unit mass, so a = f / 1.0.
    f = compute_wall_forces(p0)
    f += compute_agent_forces(id, p)
    f += compute_driving_force(id, p, v, g, dt)
    a[id] = f / mass


@wp.kernel
def integrate(p: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
              a: wp.array(dtype=wp.vec3), dt: float):
    id = wp.tid()
    p[id] += v[id] * dt
    v[id] += a[id] * dt
    # Clamp speed; it doesn't get to accelerate indefinitely.
    new_speed = wp.norm_l2(v[id])
    if new_speed > max_speed:
        v[id] = v[id] * (max_speed / new_speed)


@wp.kernel
def update_box_density(p: wp.array(dtype=wp.vec3), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    rho_support = rho_kernel_size
    kSize = int(rho_support * float(field_resolution) / domain_size + 0.5)
    field_size = float(field_resolution)
    kDelta = kSize // 2
    cell_population = 1.0 / float(rho_support * rho_support)
    for a in range(len(p)):
        pos = p[a]
        # Map position to field coordinates.
        x = int(((pos.x + (domain_size * 0.5)) / domain_size) * field_size + 0.5)
        y = int(((pos.y + (domain_size * 0.5)) / domain_size) * field_size + 0.5)
        # Splat each agent into a kxk block.
        if x >= i - kDelta and x <= i + kDelta and y >= j - kDelta and y <= j + kDelta:
            rho[j, i] += cell_population


@wp.kernel
def update_circle_density(p: wp.array(dtype=wp.vec3), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    field_size = float(field_resolution)
    cx = (float(i) + 0.5) * domain_size / field_size - (domain_size * 0.5)
    cy = (float(j) + 0.5) * domain_size / field_size - (domain_size * 0.5)
    c = wp.vec3(cx, cy, 0.0)
    rho_support = rho_kernel_size / 2.0
    support_sq = rho_support * rho_support
    cell_population = 1.0 / (np.pi * support_sq)
    for a in range(len(p)):
        pos = p[a]
        dist_sq = wp.length_sq(pos - c)
        if dist_sq <= support_sq:
            rho[j, i] += cell_population


@wp.kernel
def update_first_order_density(p: wp.array(dtype=wp.vec3), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    field_size = float(field_resolution)
    cx = (float(i) + 0.5) * domain_size / field_size - (domain_size * 0.5)
    cy = (float(j) + 0.5) * domain_size / field_size - (domain_size * 0.5)
    c = wp.vec3(cx, cy, 0.0)
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
def update_gauss_density(p: wp.array(dtype=wp.vec3), rho: wp.array2d(dtype=float)):
    i, j = wp.tid()
    rho[j, i] = 0.0
    cx = (float(i) + 0.5) * domain_size / float(field_resolution) - (domain_size * 0.5)
    cy = (float(j) + 0.5) * domain_size / float(field_resolution) - (domain_size * 0.5)
    c = wp.vec3(cx, cy, 0.0)
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

def random_scenario(num_agents: int, rng: np.random.Generator):
    # TODO: This doesn't start in a collision-free configuration. That's
    # problematic.
    # Initial positions randomly distributed through the domain.
    measure = domain_size - (radius * 2.0)
    positions = (rng.random((num_agents, 3)) - 0.5) * measure
    velocities = np.zeros_like(positions)
    goals = -positions

    # Random colors.
    colors = rng.random((num_agents, 3))

    return Scenario(positions, velocities, goals, colors)


def circle_scenario(num_agents: int, rng: np.random.Generator):
    """Configure one or more concentric rings with the agents distributed
    evenly around the circumference. It fills from outside, inward."""
    # The minimum distance between agents on the same ring. It should never
    # be smaller than 2 * radius (because that would lead to collisions).
    spacing = radius * 6.0
    # Given the spacing, the minimum radius is the circle with six agents.
    min_R = spacing * 6 / (2.0 * np.pi)
    # The maximum radius is the largest circle that can fit in the domain with
    # some padding to avoid agents starting with non-negligible wall forces.
    max_R = domain_size * 0.5 - (radius * 6.0)
    # delta_R is the approximate change in radius for each ring.
    delta_R = spacing

    # We need to know how many rings to fit the requested number of agents.
    ring_count = 0
    remaining_agents = num_agents
    while remaining_agents > 0:
        R_i = max_R - ring_count * delta_R
        if R_i < min_R:
            break
        ring_capacity = int(2.0 * np.pi * R_i / spacing + 0.5)
        remaining_agents -= ring_capacity
        ring_count += 1
    agent_capacity = num_agents - remaining_agents
    if agent_capacity < num_agents:
        print(f"Warning: Reducing number of agents from {num_agents} to "
              f"{agent_capacity} to fit in circle scenario.")
        num_agents = agent_capacity
    print(f"Populating with {num_agents} agents in {ring_count} rings.")

    # Add a little noise to break symmetry.
    positions = (rng.random((num_agents, 3)) - 0.5) * 0.75 * radius
    positions[:, 2] = 0.0
    velocities = np.zeros_like(positions)
    goals = np.empty_like(positions)
    colors = np.empty((num_agents, 3), dtype=np.float32)

    i = 0
    for r_i in range(ring_count):
        R_i = max_R - r_i * delta_R
        ring_capacity = int(2.0 * np.pi * R_i / spacing + 0.5)
        ring_agents = min(ring_capacity, num_agents - i)
        dtheta = 2.0 * np.pi / ring_agents
        for j in range(ring_agents):
            theta = j * dtheta
            c = np.cos(theta)
            s = np.sin(theta)
            positions[i, 0] += R_i * c
            positions[i, 1] += R_i * s
            goals[i, :] = -positions[i, :]
            colors[i, :] = np.array([c * 0.5 + 0.5, s * 0.5 + 0.5, 0.5])
            i += 1
    if np.max(colors) > 1.0 or np.min(colors) < 0.0:
        too_small = (colors < 0.0).nonzero()
        too_big = (colors > 1.0).nonzero()
        print("Color error!", too_small, too_big)
    return Scenario(positions, velocities, goals, colors)


def four_blocks_scenario(num_agents: int, rng: np.random.Generator):
    """The random number generator (rng) is currently not used, but the
    argument is provided for compatibility with the other scenario functions.
    """
    print(f"Warning: Four blocks ignores the number of agents argument.")

    num_agents = 100 # 4 blocks of 25 agents each.
    # Add a little noise to break symmetry.
    positions = (rng.random((num_agents, 3)) - 0.5) * 2 * radius
    positions[:, 2] = 0.0
    velocities = np.empty_like(positions)
    goals = np.empty_like(positions)
    colors = np.empty((num_agents, 3), dtype=np.float32)

    # Block measure is 3/4 of a quadrant.
    w = domain_size * 0.5 * 0.75
    dx = w / 4.0
    p0 = domain_size * 0.5 * 0.125
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
                positions[i, 0] += x0 + ix * dx
                positions[i, 1] += y0 + iy * dy
                goals[i, :] = -positions[i, :]
                velocities[i, :] = 0.0
                colors[i, :] = c
                i += 1
    return Scenario(positions, velocities, goals, colors)


class Simulation:
    def __init__(self, scenario: Scenario, use_grid: bool, timing: bool = False,
                 exit_on_stop: bool = False):
        self.num_agents = len(scenario.positions)
        self.positions = wp.array(scenario.positions, dtype=wp.vec3)
        self.velocities = wp.array(scenario.velocities, dtype=wp.vec3)
        self.accels = wp.zeros_like(self.velocities)

        self.goals = wp.array(scenario.goals, dtype=wp.vec3)

        self.use_grid = use_grid
        if self.use_grid:
            grid_rez = 32
            self.grid = wp.HashGrid(grid_rez, grid_rez, 1)
            self.grid_cell_size = domain_size / grid_rez

        self.agent_radius = radius

        self.density_kernel = scenario.density_kernel
        self.density = wp.zeros((field_resolution, field_resolution), dtype=float)

        # TODO: Consider setting dt and inferring the number of substeps so
        # that the solve gets a value on the order of 0.001. That might be more
        # intuitive.
        # Note: the bigger the number of substeps, the faster things go,
        # because we do fewer density field computations (one for ever N sub
        # steps).
        self.sub_steps = 50
        self.dt = 0.001 * self.sub_steps  # For each solve, dt = 0.001.
        self.step_count = 0

        self.show_timings = timing
        self.exit_on_stop = exit_on_stop

    def validate_state(self, i: int):
        v = self.velocities.numpy()
        speeds = np.linalg.norm(v[:, :2], axis=1)
        if not (speeds.max() <= max_speed * 1.1):
            raise RuntimeError(
                        f"Warning: Step {self.step_count}.{i}: "
                        f"speed {speeds.max():.2f} is invalid. Should lie in "
                        f"the range [0, {max_speed:.2f}]! "
                        "Consider reducing dt or increasing max_speed.")

    def step(self):
        self.step_count += 1
        self.validate_state(-1)
        with wp.ScopedTimer("step", active=self.show_timings):
            sub_dt = self.dt / self.sub_steps
            for i in range(self.sub_steps):
                if self.use_grid:
                    self.grid.build(self.positions, self.grid_cell_size)
                    wp.launch(compute_accel_grid, dim=self.num_agents,
                            inputs=[self.positions, self.velocities, self.accels,
                                    self.goals, sub_dt, self.grid.id]
                    )
                else:
                    wp.launch(compute_accel, dim=self.num_agents,
                            inputs=[self.positions, self.velocities, self.accels,
                                    self.goals, sub_dt]
                    )

                wp.launch(integrate, dim=self.num_agents,
                        inputs=[self.positions, self.velocities, self.accels,
                                sub_dt]
                )
                self.validate_state(i)
            disp = self.positions.numpy() - self.goals.numpy()
            dist_sq = np.sum(disp[:, :2] * disp[:, :2], axis=1)
            if (dist_sq < radius * 0.25).all():
                print(f"All agents have reached their goals at step "
                      f"{self.step_count}!")
                if self.exit_on_stop:
                    sys.exit()
                return False
        return True

    def update_density(self):
        wp.launch(self.density_kernel, dim=(field_resolution, field_resolution),
                  inputs=[self.positions, self.density])

    def map_to_field(self, p):
        """Given a position in "world" space, map it to the density field.

        [-hw, -hhw]x[hw, hh] --> [0, 0]x[field_resolution, field_resolution]
        """
        p += np.array(((domain_size * 0.5, domain_size * 0.5, 0.0),))
        p *= field_resolution / domain_size
        return p

    def step_and_render_frame(self, frame_num=None, agents=None, density_img=None):
        running = self.step()
        
        # Update agent patches
        with wp.ScopedTimer("render", active=self.show_timings):
            if agents:
                positions = self.map_to_field(self.positions.numpy())
                for i, agent in enumerate(agents):
                    pos = positions[i]
                    agent.center = (pos[0], pos[1])
        if density_img:
            with wp.ScopedTimer("density", active=self.show_timings):
                self.update_density()
                rho = self.density.numpy()
                density_img.set_array(rho)
                if self.show_timings:
                    print(f"Total population = {(cell_area * rho.sum()):.2f}")

        if not running:
            global seq
            if seq is not None:
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
                        default='gauss',
                        help=f"Choose a density-field kernel: {', '.join(kernels.keys())}")
    parser.add_argument('--timing', action='store_true',
                        help="Enable timing output.")
    parser.add_argument('--use_grid', action='store_true',
                        help="Use a spatial hash grid for neighbor queries.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument('--exit_on_stop', action='store_true',
                        help="Exit the program when all agents have reached their goals.")

    # Simulation constants.
    constants = (
        ('domain_size', float, domain_size, 'The size of the square simulation domain in meters'),
        ('radius', float, radius, 'The radius of each agent in meters'),
        ('neighbor_distance', float, neighbor_distance, 'The distance within which agents consider neighbors'),
        ('pref_speed', float, pref_speed, 'The preferred speed of agents in meters per second'),
        ('max_speed', float, max_speed, 'The maximum speed of agents in meters per second'),
        ('mass', float, mass, 'The mass of each agent in kilograms'),
        ('agent_scale', float, agent_scale, 'Scaling factor for agent forces'),
        ('obstacle_scale', float, obstacle_scale, 'Scaling factor for obstacle forces'),
        ('reaction_time', float, reaction_time, 'Reaction time of agents in seconds'),
        ('force_distance', float, force_distance, 'Distance over which forces act'),
        ('field_resolution', int, field_resolution, 'Resolution of the density field'),
        ('rho_kernel_size', float, rho_kernel_size, 'The width of the density kernel support in meters'),
    )
    for name, type_, default_val, help in constants:
        parser.add_argument(f'--{name}', type=type_, default=default_val, help=f'{help}; defaults to {default_val}.')

    args = parser.parse_args()
    configure_constants(**vars(args))

    rng = np.random.default_rng(args.seed)
    scenario = scenarios[args.scenario](args.num_agents, rng)

    with wp.ScopedDevice(args.device):
        import matplotlib
        import matplotlib.patches as patches
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        scenario.density_kernel = kernels[args.density]
        sim = Simulation(scenario, args.use_grid, args.timing, args.exit_on_stop)

        agents = []

        fig, ax = plt.subplots(figsize=(12, 12))

        img = plt.imshow(
            sim.density.numpy(),
            origin="lower",
            animated=True,
            interpolation="antialiased",
        )
        img.set_norm(matplotlib.colors.Normalize(0.0, 6.0))
        plt.colorbar(img, label='ρ (people/m²)')

        # Change the axis ticks to be simulation world coordinates.
        ticks = [0, field_resolution * 0.25, field_resolution * 0.5, field_resolution * 0.75, field_resolution]
        labels = [-domain_size * 0.5, -domain_size * 0.25, 0.0, domain_size * 0.25, domain_size * 0.5]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.set_aspect('equal') # Important for circles to appear round

        # Add circles as patches.
        # We're scaling the radius to the field dimensions. We're assuming that
        # the field and simulation domain have the same aspect ratio.
        r = sim.agent_radius * field_resolution / domain_size
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

        