import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

#Next challenge, get maneuver nodes into new solver
#Arbitrary precision arithmetic
#Use events

def plot(data_values):

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  q = np.linspace(0, 2* np.pi, 100)
  v = np.linspace(0, np.pi, 100)

  x = 6378 * np.outer(np.cos(q), np.sin(v))
  y = 6378 * np.outer(np.sin(q), np.sin(v))
  z = 6356 * np.outer(np.ones(np.size(q)), np.cos(v))
  ax.plot_surface(x, y, z, rstride=4, cstride=4, color="r", alpha = 0.5)

  labels = []

  for i in range(len(data_values)):
    ax.plot(data_values[i][0, :], data_values[i][1, :], data_values[i][2, :])
    labels.append(str(i+1))

  ax.legend(labels)

  plt.show()

#This is the previous event code for trying to use it the event way

# def event(t, y):
#   global t_maneuver
#   global list_time_mnodes
#   global list_values_mnodes
#   list_time_mnodes = list_time_mnodes[1:]
#   check = int(t)
#   print(check)
#   if check in t_maneuver.keys():
#     loc = list_time_mnodes.index(check)
#     temp = list_values_mnodes[loc]
#     if temp[0] == "pr":
#       y[3:] = prograde(temp[1], y[3:])
#
#     elif temp[0] == "re":
#       y[3:] = retrograde(temp[1], y[3:])
#
#     elif temp[0] == "no":
#       y[3:] = normal(temp[1], y[3:], y[:3])
#
#     elif temp[0] == "an":
#       y[3:] = anti_normal(temp[1], y[3:], y[:3])
#
#     elif temp[0] == "ri":
#       y[3:] = radial_in(temp[1], y[3:], y[:3])
#
#     elif temp[0] == "ro":
#       y[3:] = radial_out(temp[1], y[3:], y[:3])
#
#     else:
#       print("You need to enter a valid direction")
#
#
#   return min(list_time_mnodes)-t

def prograde(dV, V):
  Vx = V[0]
  Vy = V[1]
  Vz = V[2]
  mag = (Vx**2 + Vy**2 + Vz**2)**0.5
  x = dV * Vx / mag
  y = dV * Vy / mag
  z = dV * Vz / mag
  return V + np.array([x, y, z])

def retrograde(dV, V):
  Vx = V[0]
  Vy = V[1]
  Vz = V[2]
  mag = (Vx**2 + Vy**2 + Vz**2)**0.5
  x = -dV * Vx / mag
  y = -dV * Vy / mag
  z = -dV * Vz / mag
  return V + np.array([x, y, z])

def normal(dV, V, R):
  #out of the screen is defined as normal
  Vx = V[0]
  Vy = V[1]
  Vz = V[2]
  Rx = R[0]
  Ry = R[1]
  Rz = R[2]
  cross = np.array([Vy*Rz-Vz*Ry, -Vx*Rz+Vz*Rx, Vx*Ry-Vy*Rx])
  mag = (cross[0]**2 + cross[1]**2 + cross[2]**2)**0.5
  return V + np.array([dV*cross[0]/mag, dV*cross[1]/mag, dV*cross[2]/mag])

def anti_normal(dV, V, R):
  #into the screen is defined as anti normal
  Vx = V[0]
  Vy = V[1]
  Vz = V[2]
  Rx = R[0]
  Ry = R[1]
  Rz = R[2]
  cross = np.array([Vy*Rz-Vz*Ry, -Vx*Rz+Vz*Rx, Vx*Ry-Vy*Rx])
  mag = (cross[0]**2 + cross[1]**2 + cross[2]**2)**0.5
  return V + np.array([-dV*cross[0]/mag, -dV*cross[1]/mag, -dV*cross[2]/mag])

def radial_in(dV, V, R):
  Vx = V[0]
  Vy = V[1]
  Vz = V[2]
  Rx = R[0]
  Ry = R[1]
  Rz = R[2]
  normal = np.array([Vy*Rz-Vz*Ry, -Vx*Rz+Vz*Rx, Vx*Ry-Vy*Rx])
  cross = np.array([normal[1]*Vz - normal[2]*Vy, -normal[0]*Vz + normal[2]*Vx, normal[0]*Vy - normal[1]*Vx])
  mag = (cross[0]**2 + cross[1]**2 + cross[2]**2)**0.5
  return V + np.array([dV*cross[0]/mag, dV*cross[1]/mag, dV*cross[2]/mag])

def radial_out(dV, V, R):
  Vx = V[0]
  Vy = V[1]
  Vz = V[2]
  Rx = R[0]
  Ry = R[1]
  Rz = R[2]
  normal = np.array([Vy*Rz-Vz*Ry, -Vx*Rz+Vz*Rx, Vx*Ry-Vy*Rx])
  cross = np.array([normal[1]*Vz - normal[2]*Vy, -normal[0]*Vz + normal[2]*Vx, normal[0]*Vy - normal[1]*Vx])
  mag = (cross[0]**2 + cross[1]**2 + cross[2]**2)**0.5
  return V + np.array([-dV*cross[0]/mag, -dV*cross[1]/mag, -dV*cross[2]/mag])


#Here is the function code. I tried to put the maneuver inside of the function and using global variables and other functions, however, the t doesn't change linearly and for some reason I can't update the r vectors properly
def f(t, r):


  global t_maneuver
  global list_time_mnodes
  global list_values_mnodes
  check = int(t)

  # if check in t_maneuver.keys():
  #   loc = list_time_mnodes.index(check)
  #   temp = list_values_mnodes[loc]
  #   print(r)
  #   if temp[0] == "pr":
  #     r[3:] = prograde(temp[1], r[3:])
  #     print(r)
  #
  #   elif temp[0] == "re":
  #     r[3:] = retrograde(temp[1], r[3:])
  #     print(r)
  #
  #   elif temp[0] == "no":
  #     r[3:] = normal(temp[1], r[3:], r[:3])
  #     print(r)
  #
  #   elif temp[0] == "an":
  #     r[3:] = anti_normal(temp[1], r[3:], r[:3])
  #     print(r)
  #
  #   elif temp[0] == "ri":
  #     r[3:] = radial_in(temp[1], r[3:], r[:3])
  #     print(r)
  #
  #   elif temp[0] == "ro":
  #     r[3:] = radial_out(temp[1], r[3:], r[:3])
  #     print(r)
  #
  #   else:
  #     print("You need to enter a valid direction")


  x = r[0]
  y = r[1]
  z = r[2]
  xp = r[3]
  yp = r[4]
  zp = r[5]

  mag_r = (x**2 + y**2 + z**2)**0.5

  xpp = (-GM/mag_r**3) * x
  ypp = (-GM/mag_r**3) * y
  zpp = (-GM/mag_r**3) * z


  return xp, yp, zp, xpp, ypp, zpp




if __name__ == "__main__":
  GM = 3.986e5


  N = 100000
  # pr is prograde, re is retrograde, no is normal, an is anti normal, ri is radial in, ro is radial out, the other value in the list is the magnitude of the velocity vector being added (delta V)
  t_maneuver = {750*8:["pr", 0], 1250*8:["no", 0.5], 4150*8:["an", 0.5], 6350*8:["pr", 2], 11350*8:["ri", 2], 95000:["re", 2]}
  list_time_mnodes = list(t_maneuver.keys())
  list_values_mnodes = list(t_maneuver.values())

  u = np.zeros((6, N))

  t = np.zeros(N)

  u[:, 0] = np.array([8800, 0, 0, 0 , 7.6 , 0])

  t[0] = 0

  h = 10

  data_values = []
  for i in range(len(list_time_mnodes)+1):

    if i == 0:
      sol = integrate.solve_ivp(f, (0, list_time_mnodes[i]), u[:, 0], "DOP853", t_eval = np.linspace(0,list_time_mnodes[i],301))
      data_values.append(sol.y)
      x, y, z, xs, ys, zs = sol.y
      cur_values = np.array([x[-1], y[-1], z[-1], xs[-1], ys[-1], zs[-1]])


    else:
      if i == len(list_time_mnodes):
        time_gap = N - list_time_mnodes[i-1]
      else:
        time_gap = list_time_mnodes[i] - list_time_mnodes[i-1]


      temp = list_values_mnodes

      if temp[i-1][0] == "pr":
        cur_values[3:] = prograde(temp[i-1][1], cur_values[3:])

      elif temp[i-1][0] == "re":
        cur_values[3:] = retrograde(temp[i-1][1], cur_values[3:])

      elif temp[i-1][0] == "no":
        cur_values[3:] = normal(-(temp[i-1][1]), cur_values[3:], cur_values[:3])

      elif temp[i-1][0] == "an":
        cur_values[3:] = anti_normal(-(temp[i-1][1]), cur_values[3:], cur_values[:3])

      elif temp[i-1][0] == "ri":
        cur_values[3:] = radial_in(temp[i-1][1], cur_values[3:], cur_values[:3])

      elif temp[i-1][0] == "ro":
        cur_values[3:] = radial_out(temp[i-1][1], cur_values[3:], cur_values[:3])

      else:
        print("You need to enter a valid direction")


      sol = integrate.solve_ivp(f, (0, time_gap), cur_values, "DOP853", t_eval = np.linspace(0, time_gap,51))
      data_values.append(sol.y)
      x, y, z, xs, ys, zs = sol.y
      cur_values = np.array([x[-1], y[-1], z[-1], xs[-1], ys[-1], zs[-1]])


  plot(data_values)


  # for n in range(0, N-1):
  #
  #   if t[n] in t_maneuver.keys():
  #     loc = list_time_mnodes.index(t[n])
  #     temp = list_values_mnodes[loc]
  #     if temp[0] == "pr":
  #       u[3:, n] = prograde(temp[1], u[3:, n])
  #
  #     elif temp[0] == "re":
  #       u[3:, n] = retrograde(temp[1], u[3:, n])
  #
  #     elif temp[0] == "no":
  #       u[3:, n] = normal(temp[1], u[3:, n], u[:3, n])
  #
  #     elif temp[0] == "an":
  #       u[3:, n] = anti_normal(temp[1], u[3:, n], u[:3, n])
  #
  #     elif temp[0] == "ri":
  #       u[3:, n] = radial_in(temp[1], u[3:, n], u[:3, n])
  #
  #     elif temp[0] == "ro":
  #       u[3:, n] = radial_out(temp[1], u[3:, n], u[:3, n])
  #
  #     else:
  #       print("You need to enter a valid direction")
  #
  #   mag_r = np.linalg.norm(u[:3, n])
  #   k1 = np.array([
  #                   u[3, n],
  #                   u[4, n],
  #                   u[5, n],
  #                   (-GM/mag_r**3) * u[0, n],
  #                   (-GM/mag_r**3) * u[1, n],
  #                   (-GM/mag_r**3) * u[2, n]])
  #
  #   temp_u_mid = u[:, n] + h / 2 * k1
  #
  #   mag_r = np.linalg.norm(temp_u_mid[:3])
  #   k2 = np.array([
  #                   temp_u_mid[3],
  #                   temp_u_mid[4],
  #                   temp_u_mid[5],
  #                   (-GM/mag_r**3) * temp_u_mid[0],
  #                   (-GM/mag_r**3) * temp_u_mid[1],
  #                   (-GM/mag_r**3) * temp_u_mid[2]])
  #
  #
  #   temp_u_mid = u[:, n] + h / 2 * k2
  #   mag_r = np.linalg.norm(temp_u_mid[:3])
  #   k3 = np.array([
  #                   temp_u_mid[3],
  #                   temp_u_mid[4],
  #                   temp_u_mid[5],
  #                   (-GM/mag_r**3) * temp_u_mid[0],
  #                   (-GM/mag_r**3) * temp_u_mid[1],
  #                   (-GM/mag_r**3) * temp_u_mid[2]])
  #
  #   temp_u_mid = u[:, n] + h * k3
  #   mag_r = np.linalg.norm(temp_u_mid[:3])
  #   k4 = np.array([
  #                   temp_u_mid[3],
  #                   temp_u_mid[4],
  #                   temp_u_mid[5],
  #                   (-GM/mag_r**3) * temp_u_mid[0],
  #                   (-GM/mag_r**3) * temp_u_mid[1],
  #                   (-GM/mag_r**3) * temp_u_mid[2]])
  #
  #
  #   u[:, n+1] = u[:, n] + h * (k1 + 2 * k2 + 2 * k3 + k4)/6
  #
  #   t[n+1] = t[n] + h

  #plot(u)

