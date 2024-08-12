import cv2 
import numpy as np

webcam = cv2.VideoCapture(1) #qual camera usar
width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))   #pega a largura
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)) #pega a altura

# Define as tralhas da linha horizontal
linha_y = height // 2
linha_inicial = 120
linha_final = 520
grossura_linha = 10
linha_largura = linha_final - linha_inicial

# Define as tralhas da linha vertical
linha_x = width // 2
linha_top = 100  # Início da linha vertical
linha_bottom = (height // 4) + 50 # Fim da linha vertical
grossura_linha_vertical = 10
linha_largura_vertical = linha_bottom - linha_top

#                                               rois horizontais
num_rois_horizontal = 4
roi_fraction_horizontal = 1.0 / num_rois_horizontal
roi_width_horizontal = int(linha_largura * roi_fraction_horizontal)
roi_posicoes_horizontal = [linha_inicial + int(i * roi_width_horizontal) for i in range(num_rois_horizontal)]

#                                               rois verticais
num_rois_vertical = 1
roi_fraction_vertical = 1.0 / num_rois_vertical
roi_width_vertical = int(linha_largura_vertical * roi_fraction_vertical)
roi_posicoes_vertical = [linha_top + int(i * roi_width_vertical) for i in range(num_rois_vertical)]


while True:
    camera, frame = webcam.read()
    filmagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     
    frame = cv2.line(frame, (linha_inicial, linha_y), (linha_final, linha_y), (255, 0, 0), grossura_linha)  #Linha horizontal
    frame = cv2.line(frame, (linha_x, linha_top), (linha_x, linha_bottom), (255, 0, 0), grossura_linha_vertical) # linha frente em cima
    frame = cv2.line(frame, ((width // 2), 0), ((width // 2), height), (0, 255, 0), 1) #linha vertical guia
    #COMO LER A LINHA 
    #A imagem onde a linha deve ser desenhada.
    #O ponto de partida com as coordenadas xey.
    #O ponto final com as coordenadas xey.
    #Na próxima propriedade, estamos atribuindo a cor à linha.  Aqui, o formato é BGR.
    #define a espessura da linha.


    statuses_horizontal = []
    output_rois_horizontal = []


    
    for i, roi_left in enumerate(roi_posicoes_horizontal):
        roi_right = min(width, roi_left + roi_width_horizontal)
        roi_top = linha_y - (grossura_linha // 2)
        roi_bottom = linha_y + (grossura_linha // 2)

        # Ajusta os limites dos ROIs
        roi_top = max(0, roi_top)
        roi_bottom = min(height, roi_bottom)
        roi_left = max(0, roi_left)
        roi_right = min(width, roi_right)

        roi_horizontal = filmagem_cinza[roi_top:roi_bottom, roi_left:roi_right]
        mean_pixel_value_horizontal = np.mean(roi_horizontal)

        threshold = 145
        status_horizontal = mean_pixel_value_horizontal < threshold

        statuses_horizontal.append(status_horizontal)
        output_rois_horizontal.append(f"ROI Hor. {i + 1}: {mean_pixel_value_horizontal:.2f}")

        if status_horizontal:
            cv2.putText(frame, f"Detectado. {i + 1}", (25 + (i * 200), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            color = (0, 255, 255)  # Amarelo
            frame = cv2.line(frame, (roi_left, linha_y), (roi_right, linha_y), color, grossura_linha - 5)

    # Análise da linha vertical
    statuses_vertical = []
    output_rois_vertical = []

    for i, roi_top in enumerate(roi_posicoes_vertical):
        roi_bottom = min(height, roi_top + roi_width_vertical)
        roi_left = linha_x - (grossura_linha_vertical // 2)
        roi_right = linha_x + (grossura_linha_vertical // 2)

        # Ajusta os limites dos ROIs
        roi_left = max(0, roi_left)
        roi_right = min(width, roi_right)
        roi_top = max(0, roi_top)
        roi_bottom = min(height, roi_bottom)

        roi_vertical = filmagem_cinza[roi_top:roi_bottom, roi_left:roi_right]
        mean_pixel_value_vertical = np.mean(roi_vertical)

        threshold = 150
        status_vertical = mean_pixel_value_vertical < threshold

        statuses_vertical.append(status_vertical)
        output_rois_vertical.append(f"ROI 4. {i + 1}: {mean_pixel_value_vertical:.2f}")

        if status_vertical:
            cv2.putText(frame, f"Detectado 5", (25 + (i * 200), height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            color = (0, 255, 255)
            frame = cv2.line(frame, (linha_x, roi_top), (linha_x, roi_bottom), color, grossura_linha_vertical - 5)

    # Exibe os valores médios dos ROIs no console
    print(", ".join(output_rois_horizontal), ", ".join(output_rois_vertical))
    

    # Exibe a imagem com as anotações
    cv2.imshow("Imagem", frame)
           # print(statuses)

    #print(", ".join(output_rois))
    
    cv2.imshow("Imagem", frame)
    if cv2.waitKey(1) == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()