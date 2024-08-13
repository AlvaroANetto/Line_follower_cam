import random
erro= random.randint(0, 100) - 50
integral = erro
integral = integral + erro
ultimo_erro = erro
derivada = erro - ultimo_erro
kd = derivada * 1
kp = erro * 0.3
ki = integral * 0.006

for i in range(0, 10):
    erro = random.randint(0, 100) - 50
    print(kd, "|", kp, "|", ki)
