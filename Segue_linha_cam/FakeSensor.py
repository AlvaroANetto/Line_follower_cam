import cv2
import numpy as np
import time

# Abre a câmera
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Dimensões do vídeo
width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Linha horizontal
linha_y = height // 2
linha_inicial, linha_final = 165, 450
grossura_linha = 10
linha_largura = linha_final - linha_inicial

# ROIs horizontais
num_rois_horizontal = 7
roi_width_horizontal = linha_largura // num_rois_horizontal
roi_posicoes_horizontal = [linha_inicial + i * roi_width_horizontal for i in range(num_rois_horizontal)]
threshold_horizontal = 120

# PID
integral = 0
ultimo_erro = 0
kp_gain = 0.3
ki_gain = 0.006
kd_gain = 1

while True:
    start_time = time.time()
    ret, frame = webcam.read()
    if not ret:
        print("Erro ao capturar a imagem.")
        break

    filmagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Desenha linha de referência
    cv2.line(frame, (linha_inicial, linha_y), (linha_final, linha_y), (255, 0, 0), grossura_linha)

    # Análise dos ROIs horizontais
    pesos = []
    intensidades = []

    for i, roi_left in enumerate(roi_posicoes_horizontal):
        roi_right = roi_left + roi_width_horizontal
        roi_top = max(0, linha_y - grossura_linha // 2)
        roi_bottom = min(height, linha_y + grossura_linha // 2)

        roi_horizontal = filmagem_cinza[roi_top:roi_bottom, roi_left:roi_right]
        mean_val = np.mean(roi_horizontal)
        intensidades.append(mean_val)

        # Peso de posição (-3, -2, -1, 0, 1, 2, 3)
        peso = i - (num_rois_horizontal // 2)
        pesos.append(peso)

        # Visualização
        if mean_val < threshold_horizontal:
            cv2.putText(frame, f"ROI {i+1}", (25 + (i * 100), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.line(frame, (roi_left, linha_y), (roi_right, linha_y), (0, 255, 255), grossura_linha - 5)

    # Cálculo do erro (média ponderada)
    soma_pesos = 0
    soma_ativos = 0
    for peso, intensidade in zip(pesos, intensidades):
        if intensidade < threshold_horizontal:
            soma_pesos += peso
            soma_ativos += 1

    if soma_ativos > 0:
        erro = soma_pesos / soma_ativos
    else:
        erro = 0  # Nenhuma linha detectada

    # PID
    integral += erro
    derivada = erro - ultimo_erro
    ultimo_erro = erro

    kp = kp_gain * erro
    ki = ki_gain * integral
    kd = kd_gain * derivada

    pid_saida = kp + ki + kd

    #print(f"Erro: {erro:.2f} | PID: {pid_saida:.2f} (Kp:{kp:.2f}, Ki:{ki:.2f}, Kd:{kd:.2f})")
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  
    print(f"Tempo por frame: {elapsed_time:.2f} ms")

    cv2.imshow("Imagem", frame)

    if cv2.waitKey(1) == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
