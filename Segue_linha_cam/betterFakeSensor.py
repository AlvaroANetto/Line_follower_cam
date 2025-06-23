import cv2
import numpy as np
import time

# Abre a câmera
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Reduz a resolução para melhorar desempenho
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Parâmetros da linha horizontal
linha_y = height // 2
linha_inicial = 50
linha_final = width - 50
grossura_linha = 10
linha_largura = linha_final - linha_inicial

# ROIs horizontais (pré-calculados)
num_rois_horizontal = 5
roi_fraction_horizontal = 1.0 / num_rois_horizontal
roi_width_horizontal = int(linha_largura * roi_fraction_horizontal)
roi_posicoes_horizontal = []

for i in range(num_rois_horizontal):
    left = max(0, linha_inicial + int(i * roi_width_horizontal))
    right = min(width, left + roi_width_horizontal)
    top = max(0, linha_y - (grossura_linha // 2))
    bottom = min(height, linha_y + (grossura_linha // 2))
    roi_posicoes_horizontal.append((left, right, top, bottom))

while True:
    start_time = time.time()

    ret, frame = webcam.read()
    if not ret:
        break

    filmagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Desenha a linha horizontal de referência
    cv2.line(frame, (linha_inicial, linha_y), (linha_final, linha_y), (255, 0, 0), grossura_linha)

    # Análise horizontal
    for i, (left, right, top, bottom) in enumerate(roi_posicoes_horizontal):
        roi = filmagem_cinza[top:bottom:2, left:right:2]  # Amostragem para desempenho
        mean_value = np.mean(roi)
        if mean_value < 180:
            cv2.putText(frame, f"Hor {i+1}", (25 + i * 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(frame, (left, linha_y), (right, linha_y), (0, 255, 255), grossura_linha - 5)

    # Tempo de execução por frame
    end_time = time.time()
    frame_time_ms = (end_time - start_time) * 1000
    print(f"Frame time: {frame_time_ms:.2f} ms")

    cv2.imshow("Imagem", frame)
    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
