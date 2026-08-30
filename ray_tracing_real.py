import numpy as np
import matplotlib.pyplot as plt

### Valeurs réelles des constantes (Système International)
G = 6.67430e-11  # m^3 kg^-1 s^-2
M = 1.989e30     # kg (Masse du soleil)
c = 299792458.0  # m/s (Vitesse de la lumière)

Rs = 2 * G * M / c**2

res = 350
# Mise à l'échelle spatiale (équivalent à D=20 et taille=15 quand Rs=2)
D = 10 * Rs
taille_ecran = 7.5 * Rs 

x_ecran = np.linspace(-taille_ecran, taille_ecran, res)
z_ecran = np.linspace(-taille_ecran, taille_ecran, res)
X, Z = np.meshgrid(x_ecran, z_ecran)

x = X.flatten()
angle_camera = np.radians(5) 

y_cam = -D * np.cos(angle_camera)
z_cam = D * np.sin(angle_camera)

y = np.full_like(x, y_cam)
z = np.full_like(x, z_cam) + Z.flatten()

# CORRECTION : Les rayons sont de la lumière, on multiplie par c
vx = np.zeros_like(x)
vy = c * np.cos(angle_camera) * np.ones_like(x)
vz = -c * np.sin(angle_camera) * np.ones_like(x)

image = np.zeros_like(x)

# CORRECTION : Le pas de temps dépend du temps de traversée du trou noir par la lumière
dt = 0.05 * (Rs / c) 
etapes = 400

Lx = y * vz - z * vy
Ly = z * vx - x * vz
Lz = x * vy - y * vx
L2 = Lx**2 + Ly**2 + Lz**2

for i in range(etapes):
    r = np.sqrt(x**2 + y**2 + z**2)
    # Les frontières de calcul s'adaptent à Rs
    actif = (r > Rs) & (r < 17.5 * Rs)

    r_secu = np.maximum(r, 0.05 * Rs)
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

    r_disque = np.sqrt(x**2 + y**2)
    
    # CORRECTION : L'épaisseur du disque s'adapte à Rs
    disque = (np.abs(z) < 0.2 * Rs) & (r_disque > 3*Rs) & (r_disque < 10*Rs) & actif

    if np.any(disque):
        v_rot = np.sqrt(G * M / r_disque[disque])
        # CORRECTION : Division par c dans la formule Doppler
        doppler = 1 + (v_rot / c) * (x[disque] / r_disque[disque])
        
        # On remet l'intensité à l'échelle pour éviter qu'elle soit noyée par la taille en mètres
        intensite_base = 10 / (r_disque[disque] / (Rs/2)) 
        intensite_doppler = intensite_base * (doppler ** 3)

        image[disque] += intensite_doppler * 0.1

image_2d = image.reshape((res, res))

fig = plt.figure(figsize=(8, 8), facecolor='black')
plt.imshow(image_2d, cmap='inferno', origin='lower', extent=[-taille_ecran, taille_ecran, -taille_ecran, taille_ecran])
plt.axis('off')
plt.show()