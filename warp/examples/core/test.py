import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np

# --- Setup the figure and axes ---
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal') # Ensures the circle looks like a circle, not an ellipse

agents = []
# --- Create the circle patch and add it to the axes ---
# Initial position (x, y) and radius
for i in range(5):
    circle = Circle((5, 5), 0.5, fc='blue', alpha=0.6)
    ax.add_patch(circle)
    agents.append(circle)

# --- Animation functions ---

def init():
    """Initialization function for the animation."""
    # Set the initial position (optional, can be done during creation)
    circle.center = (5, 5)
    return circle,

def animate(i, agents=None):
    """
    Update function for each frame of the animation.
    
    Args:
        i (int): The current frame number (provided by FuncAnimation).
    """
    for a, c in enumerate(agents):
        # Calculate new position based on frame number 'i' to create movement
        # For example, a simple circular motion
        offset = a * 15
        x = 5 + 3 * np.sin(np.radians((i + offset) * 2)) # Move in a circle with radius 3
        y = 5 + 3 * np.cos(np.radians((i + offset) * 2))
        
        c.center = (x, y) # Update the circle's center
    
    return agents

# --- Create and run the animation ---
# frames: number of frames in the animation (e.g., 360 frames for a smooth circle path)
# interval: delay between frames in milliseconds (e.g., 20ms)
# blit: optimizes drawing by only updating changed parts (highly recommended for performance)
ani = animation.FuncAnimation(
    fig, 
    animate,
    fargs=(agents,),
    # init_func=init, 
    frames=360, 
    interval=20, 
    blit=True
)

plt.show()

# To save the animation as a file (e.g., MP4 or GIF), you would add:
# anim.save('circle_animation.mp4', writer='ffmpeg')
