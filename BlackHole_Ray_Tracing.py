import numpy as np
import matplotlib.pyplot as plt

### General constants, two options : 
G = 1
M = 1
c = 1

### or we can use the real values of the constants, but then we have to adapt the units of the simulation (meters, seconds, kilograms)
# G = 6.67430e-11  # m^3 kg^-1 s^-2
# M = 1.989e30     # kg (mass of the Sun)
# c = 299792458    # m/s (speed of light)


Rs = 2 * G * M / c**2
print(f"Le rayon de Schwarzschild est de {Rs}")

res = 350      # Resolution of the image (number of pixels in each dimension)
D = 20         # Distance of the camera from the black hole

x_ecran = np.linspace(-15, 15, res) # Positions of the rays in the x-direction on the screen
z_ecran = np.linspace(-15, 15, res) # Positions of the rays in the z-direction on the screen
X, Z = np.meshgrid(x_ecran, z_ecran) # Create a grid of positions

x = X.flatten()
# On décale la caméra en hauteur (Z positif) et on l'incline pour regarder vers le bas
angle_camera = np.radians(5) # 25 degrés d'inclinaison

# Position initiale des rayons (caméra sur une orbite inclinée)
y_cam = -D * np.cos(angle_camera)
z_cam = D * np.sin(angle_camera)

y = np.full_like(x, y_cam)
z = np.full_like(x, z_cam) + Z.flatten()

# Directions initiales des rayons (ils avancent vers le centre en plongeant)
vx = np.zeros_like(x)
vy = np.cos(angle_camera) * np.ones_like(x)
vz = -np.sin(angle_camera) * np.ones_like(x)

image = np.zeros_like(x)

dt = 0.1
etapes = 400

### Cinetic moment for each ray (L = r x v)
Lx = y * vz - z * vy
Ly = z * vx - x * vz
Lz = x * vy - y * vx
L2 = Lx**2 + Ly**2 + Lz**2


### Principal loop for ray tracing using the Euler method

for i in range(etapes):
    r = np.sqrt(x**2 + y**2 + z**2)
    actif = (r > Rs) & (r < 35)

    r_secu = np.maximum(r, 0.1)
    A = (3 * G * M * L2) / ((c**2) * (r_secu**4))

    ax = -A[actif] * (x[actif] / r[actif])
    ay = -A[actif] * (y[actif] / r[actif])
    az = -A[actif] * (z[actif] / r[actif])

    vx[actif] = vx[actif] + ax * dt
    vy[actif] = vy[actif] + ay * dt
    vz[actif] = vz[actif] + az * dt

    x[actif] = x[actif] + vx[actif] * dt
    y[actif] = y[actif] + vy[actif] * dt
    z[actif] = z[actif] + vz[actif] * dt

    # accretion of the disk (flat disk in the plane z=0, with inner radius 3*Rs and outer radius 10*Rs)
    r_disque = np.sqrt(x**2 + y**2)
    # the disk is active if the ray is within the disk's radius and close to the plane z=0
    disque = (np.abs(z) < 0.4) & (r_disque > 3*Rs) & (r_disque < 10*Rs) & actif

    if np.any(disque):
        v_rot = np.sqrt(G * M / r_disque[disque])
        doppler = 1 + v_rot * (x[disque] / r_disque[disque])
        intensite_base = 10 / r_disque[disque]
        intensite_doppler = intensite_base * (doppler ** 3)

        # On projette l'indice 1D sur l'image globale
        # (Astuce: on filtre pour accumuler proprement)
        image[disque] += intensite_doppler * dt

### show the image
image_2d = image.reshape((res, res))

fig = plt.figure(figsize=(8, 8), facecolor='black')
plt.imshow(image_2d, cmap='inferno', origin='lower', extent=[-15, 15, -15, 15])
plt.axis('off')
plt.show()
