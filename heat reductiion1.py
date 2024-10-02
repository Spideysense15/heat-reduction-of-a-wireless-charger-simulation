import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# Define the grid size
grid_size = 30

# Generate a meshgrid for 3D plotting
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
z = np.linspace(-1, 1, grid_size)
x, y, z = np.meshgrid(x, y, z)

# Function to compute heat for specific points
def compute_heat(x, y, z):
    return np.exp(-(x**2 + y**2 + z**2))

# Function to simulate cooling mechanism
def apply_cooling(heat, factor):
    return heat * factor

# Temperature range mapping: initial (30°C) to final (12°C)
initial_temp_celsius = 30
final_temp_celsius = 12

# Compute initial and final temperatures
initial_temp = initial_temp_celsius

cooling_start_frame = 60
total_frames = 240
frames_per_second = 30  # For smoother animation

# Calculate cooling factor based on desired final temperature
cooling_factor = (final_temp_celsius / initial_temp_celsius) ** (1 / (total_frames - cooling_start_frame))

# Plot a 3D phone (simplified as a rectangular prism)
x_phone = np.array([[-0.2, -0.2, 0.2, 0.2, -0.2],
                    [-0.2, -0.2, 0.2, 0.2, -0.2],
                    [-0.2, -0.2, 0.2, 0.2, -0.2],
                    [-0.2, -0.2, 0.2, 0.2, -0.2]])
y_phone = np.array([[-0.4, 0.4, 0.4, -0.4, -0.4],
                    [-0.4, 0.4, 0.4, -0.4, -0.4],
                    [-0.4, 0.4, 0.4, -0.4, -0.4],
                    [-0.4, 0.4, 0.4, -0.4, -0.4]])
z_phone = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0]])

# Create a cooling fan (simplified as a cylindrical shape with a few blades)
def create_fan(radius=0.3, height=0.05, num_blades=4):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, height, 2)
    U, V = np.meshgrid(u, v)
    X = radius * np.cos(U)
    Y = radius * np.sin(U)
    Z = V

    # Fan blades
    blades = []
    for i in range(num_blades):
        angle = i * 2 * np.pi / num_blades
        X_blade = X * np.cos(angle) - Y * np.sin(angle)
        Y_blade = X * np.sin(angle) + Y * np.cos(angle)
        blades.append((X_blade, Y_blade, Z))
    
    return X, Y, Z, blades

X_fan, Y_fan, Z_fan, fan_blades = create_fan()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial plot setup
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_facecolor('white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.view_init(elev=20., azim=30)

# Plot initial phone surface
phone_surface = ax.plot_surface(x_phone, y_phone, z_phone, color='silver', alpha=0.8)

# Plot a 3D wireless charger (simplified as a cylinder)
u_charger = np.linspace(0, 2 * np.pi, 100)
v_charger = np.linspace(0, np.pi, 100)
x_charger = 0.6 * np.outer(np.cos(u_charger), np.sin(v_charger))
y_charger = 0.6 * np.outer(np.sin(u_charger), np.sin(v_charger))
z_charger = 0.03 * np.outer(np.ones(np.size(u_charger)), np.cos(v_charger)) - 0.7

# Function to compute scaled heat
def compute_scaled_heat(x, y, z, initial_temp, final_temp):
    heat = compute_heat(x, y, z)
    scaled_heat = heat * (initial_temp - final_temp) + final_temp
    return scaled_heat

initial_scaled_heat = compute_scaled_heat(x_charger, y_charger, z_charger, initial_temp_celsius, final_temp_celsius)

# Function to convert temperature to color
def temperature_to_color(temp, initial_temp, final_temp):
    norm_temp = (temp - final_temp) / (initial_temp - final_temp)
    return cm.coolwarm(norm_temp)

# Initialize plot surface
charger_surface = ax.plot_surface(x_charger, y_charger, z_charger, facecolors=temperature_to_color(initial_scaled_heat, initial_temp_celsius, final_temp_celsius), rstride=1, cstride=1, alpha=0.7, shade=True)

# Animation function
def update(frame):
    # Clear the axis
    ax.cla()
    
    # Re-setup the axis
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.view_init(elev=20., azim=30)
    
    if frame < cooling_start_frame:
        charger_heat = initial_scaled_heat
    else:
        frame_since_cooling_start = frame - cooling_start_frame
        charger_heat = apply_cooling(initial_scaled_heat, cooling_factor ** frame_since_cooling_start)
    
    charger_colors = temperature_to_color(charger_heat, initial_temp_celsius, final_temp_celsius)
    
    # Plot the charger and phone surfaces
    ax.plot_surface(x_charger, y_charger, z_charger, facecolors=charger_colors, rstride=1, cstride=1, alpha=0.7, shade=True)
    ax.plot_surface(x_phone, y_phone, z_phone, color='silver', alpha=0.8)
    
    # Plot the fan
    angle = frame * 2 * np.pi / total_frames
    for X_blade, Y_blade, Z_blade in fan_blades:
        X_rot = X_blade * np.cos(angle) - Y_blade * np.sin(angle)
        Y_rot = X_blade * np.sin(angle) + Y_blade * np.cos(angle)
        ax.plot_surface(X_rot, Y_rot, Z_blade - 0.75, color='gray', alpha=0.8)  # Move fan down
    
    if frame <= frames_per_second * 5:  # Display for 5 seconds
        ax.text2D(0.05, 0.95, f'Initial Temp: {initial_temp_celsius}°C', transform=ax.transAxes, fontsize=12, color='red')
    if frame >= total_frames - frames_per_second * 5:  # Display for the last 5 seconds
        ax.text2D(0.05, 0.90, f'Final Temp: {final_temp_celsius}°C', transform=ax.transAxes, fontsize=12, color='blue')

# Create animation
ani = FuncAnimation(fig, update, frames=total_frames, interval=1000 / frames_per_second, blit=False)

# Change figure background color to black
fig.patch.set_facecolor('black')

plt.show()
