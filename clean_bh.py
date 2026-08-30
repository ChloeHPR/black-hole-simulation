import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# --- Physical Constants (SI Units) ---
G_CONST = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.989e30       # Solar mass (kg)
C_SPEED = 299792458.0  # Speed of light (m/s)


class BlackHoleRenderer:
    """
    Simulates gravitational lensing and the accretion disk of a black hole 
    using ray tracing and the Euler integration method.
    """
    
    # Configuration constants relative to the Schwarzschild radius (Rs)
    DISK_INNER_RADIUS = 3.0
    DISK_OUTER_RADIUS = 10.0
    DISK_THICKNESS = 0.2
    BOUNDARY_RADIUS = 17.5
    TIME_STEP_FACTOR = 0.05
    
    def __init__(self, mass: float = M_SUN, resolution: int = 350, 
                 camera_angle_deg: float = 5.0, steps: int = 400):
        """
        Initializes the simulation parameters.
        
        Args:
            mass: Mass of the black hole in kilograms.
            resolution: The dimensions of the final image (res x res).
            camera_angle_deg: Inclination angle of the camera in degrees.
            steps: Number of integration steps for the ray tracing.
        """
        self.mass = mass
        self.resolution = resolution
        self.camera_angle_rad = np.radians(camera_angle_deg)
        self.steps = steps
        
        # Calculate Schwarzschild radius
        self.rs = 2 * G_CONST * self.mass / C_SPEED**2
        
        # Spatial scaling based on Rs
        self.camera_distance = 10 * self.rs
        self.screen_size = 7.5 * self.rs

    def _initialize_rays(self) -> Tuple[np.ndarray, ...]:
        """
        Sets up the initial position and velocity vectors for the light rays.
        
        Returns:
            A tuple containing 1D arrays for initial coordinates (x, y, z) 
            and velocities (vx, vy, vz).
        """
        # Create a grid for the screen
        x_screen = np.linspace(-self.screen_size, self.screen_size, self.resolution)
        z_screen = np.linspace(-self.screen_size, self.screen_size, self.resolution)
        x_grid, z_grid = np.meshgrid(x_screen, z_screen)
        
        x = x_grid.flatten()
        
        # Position camera on an inclined orbit
        y_cam = -self.camera_distance * np.cos(self.camera_angle_rad)
        z_cam = self.camera_distance * np.sin(self.camera_angle_rad)
        
        y = np.full_like(x, y_cam)
        z = np.full_like(x, z_cam) + z_grid.flatten()
        
        # Initial directions for the rays (plunging towards the center at speed c)
        vx = np.zeros_like(x)
        vy = C_SPEED * np.cos(self.camera_angle_rad) * np.ones_like(x)
        vz = -C_SPEED * np.sin(self.camera_angle_rad) * np.ones_like(x)
        
        return x, y, z, vx, vy, vz

    def render(self) -> np.ndarray:
        """
        Executes the main ray tracing loop.
        
        Returns:
            A 2D numpy array representing the simulated image intensities.
        """
        x, y, z, vx, vy, vz = self._initialize_rays()
        image = np.zeros_like(x)
        
        # Dynamic time step based on light crossing time
        dt = self.TIME_STEP_FACTOR * (self.rs / C_SPEED)
        
        # Calculate angular momentum for each ray (L = r x v)
        l_x = y * vz - z * vy
        l_y = z * vx - x * vz
        l_z = x * vy - y * vx
        l_squared = l_x**2 + l_y**2 + l_z**2
        
        for _ in range(self.steps):
            r = np.sqrt(x**2 + y**2 + z**2)
            
            # Mask for rays that are still active (outside horizon, inside boundary)
            active_mask = (r > self.rs) & (r < self.BOUNDARY_RADIUS * self.rs)
            
            # Avoid division by zero at the singularity
            r_safe = np.maximum(r, 0.05 * self.rs)
            
            # Acceleration magnitude derived from General Relativity metrics
            acceleration_mag = (3 * G_CONST * self.mass * l_squared) / (C_SPEED**2 * r_safe**4)
            
            # Compute acceleration vectors
            ax = -acceleration_mag[active_mask] * (x[active_mask] / r[active_mask])
            ay = -acceleration_mag[active_mask] * (y[active_mask] / r[active_mask])
            az = -acceleration_mag[active_mask] * (z[active_mask] / r[active_mask])
            
            # Euler integration for velocity and position
            vx[active_mask] += ax * dt
            vy[active_mask] += ay * dt
            vz[active_mask] += az * dt
            
            x[active_mask] += vx[active_mask] * dt
            y[active_mask] += vy[active_mask] * dt
            z[active_mask] += vz[active_mask] * dt
            
            # --- Accretion Disk Physics ---
            r_cylindrical = np.sqrt(x**2 + y**2)
            
            disk_mask = (
                (np.abs(z) < self.DISK_THICKNESS * self.rs) & 
                (r_cylindrical > self.DISK_INNER_RADIUS * self.rs) & 
                (r_cylindrical < self.DISK_OUTER_RADIUS * self.rs) & 
                active_mask
            )
            
            if np.any(disk_mask):
                # Keplerian orbital velocity
                v_rot = np.sqrt(G_CONST * self.mass / r_cylindrical[disk_mask])
                
                # Relativistic Doppler beaming
                doppler_factor = 1 + (v_rot / C_SPEED) * (x[disk_mask] / r_cylindrical[disk_mask])
                
                # Base intensity scaled down by distance
                base_intensity = 10.0 / (r_cylindrical[disk_mask] / (self.rs / 2))
                doppler_intensity = base_intensity * (doppler_factor ** 3)
                
                image[disk_mask] += doppler_intensity * 0.1
                
        # Reshape 1D array back to 2D image
        return image.reshape((self.resolution, self.resolution))


def show_image(image_data: np.ndarray, extent_limit: float):
    """
    Renders the final image data using matplotlib.
    """
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.imshow(
        image_data, 
        cmap='inferno', 
        origin='lower', 
        extent=[-extent_limit, extent_limit, -extent_limit, extent_limit]
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Configuration
    renderer = BlackHoleRenderer(resolution=350, camera_angle_deg=5.0, steps=400)
    
    # 2. Computation
    print("Tracing rays... Please wait.")
    final_image = renderer.render()
    
    # 3. Visualization
    show_image(final_image, extent_limit=renderer.screen_size)