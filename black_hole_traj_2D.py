import numpy as np
import math
import matplotlib.pyplot as plt

### on définit les constantes
G=1
M=1
C=1

Rs= 2*G*M/C**2
print(f"le rayon de schwartchild est de {Rs}")

dt = 0.05
duree = 10000

x= 5*Rs
y= 2*Rs
vx=0
vy=0.3

hx=[]
hy=[]

# Moment cinétique
L=x*vy-y*vx


for i in range(1,duree):
  r=np.sqrt(x**2+y**2)

  if r<Rs:
    print("aspiré!")
    break

  # magnitude de l'accélération attractive A
  A= (G*M/r**2)+(3*G*M*L**2)/((C**2)*(r**4))

  # decomposition de lacceleration sur les deux axes x, y
  ax=-A*(x/r)
  ay=-A*(y/r)


  # méthode d'euler
  vx=vx+ax*dt
  vy=vy+ay*dt
  x=x+vx*dt
  y=y+vy*dt


  hx.append(x)
  hy.append(y)


plt.figure(figsize=(8,8))
trou_noir=plt.Circle((0,0),Rs, color='black')
plt.gca().add_patch(trou_noir)
plt.plot(hx, hy, color='blue', label="Orbite")
plt.gca().set_aspect('equal')
plt.grid()
plt.title("trajectoire autour d'un trou noir")
plt.xlabel("Accelération selon x")
plt.ylabel("Accelération selon y")
plt.legend()
plt.show()