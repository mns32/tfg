import cv2
import time
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/pose_landmarker_full.task"

PREPARACION = "PREPARACION"
LISTO = "LISTO"
PRUEBA = "PRUEBA"

SENTADO = "SENTADO"
SUBIENDO = "SUBIENDO"
DE_PIE = "DE PIE"
BAJANDO = "BAJANDO"
COMPLETADO = "COMPLETADO"

# Colores de la interfaz en formato BGR
GRANATE = (90, 90, 190)
VERDE = (100, 190, 100)
AMARILLO = (140, 205, 220)
AZUL_GRISACEO = (205, 170, 120)
BLANCO = (235, 235, 235)
GRIS = (200, 200, 200)
NEGRO = (20, 20, 20)

fase = PREPARACION
estado = SENTADO

cadera_sentado = None
cadera_anterior = None

tiempo_inicio_estable = None
tiempo_inicio_de_pie = None
tiempo_inicio_sentado_final = None
tiempo_mensaje_listo = None

prueba_completada = False

TIEMPO_ESTABLE_INICIAL = 2.0
TIEMPO_CONFIRMAR_DE_PIE = 0.7
TIEMPO_CONFIRMAR_SENTADO = 0.7

UMBRAL_INICIO_SUBIDA = 0.035
UMBRAL_DE_PIE = 0.10
UMBRAL_SENTADO = 0.04

ANGULO_SENTADO_MAX = 135
ANGULO_DE_PIE_MIN = 155

VISIBILIDAD_MINIMA = 0.60


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)  # Abrimos la webcam

if not cap.isOpened():
    print("ERROR: No se ha podido abrir la camara.")
    landmarker.close()
    exit()


# Calcula el angulo formado por tres articulaciones
def calcular_angulo(a, b, c):
    angulo = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x)
        - math.atan2(a.y - b.y, a.x - b.x)
    )

    angulo = abs(angulo)

    if angulo > 180:
        angulo = 360 - angulo

    return angulo


# Calcula la visibilidad media de cadera, rodilla y tobillo de una pierna
def visibilidad_pierna(landmarks, lado):
    if lado == "IZQUIERDO":
        indices = [23, 25, 27]
    else:
        indices = [24, 26, 28]

    return sum(
        landmarks[i].visibility for i in indices
    ) / len(indices)


# Selecciona la pierna que MediaPipe ve con mayor claridad
def obtener_lado_visible(landmarks):
    vis_izq = visibilidad_pierna(
        landmarks,
        "IZQUIERDO"
    )

    vis_der = visibilidad_pierna(
        landmarks,
        "DERECHO"
    )

    if vis_izq >= vis_der:
        return "IZQUIERDO", vis_izq

    return "DERECHO", vis_der


# Devuelve cadera, rodilla y tobillo de la pierna seleccionada
def obtener_pierna(landmarks, lado):
    if lado == "IZQUIERDO":
        return (
            landmarks[23],
            landmarks[25],
            landmarks[27]
        )

    return (
        landmarks[24],
        landmarks[26],
        landmarks[28]
    )


# Dibuja texto centrado con un borde oscuro para separarlo del video
def texto_centrado(frame, texto, y, escala, color):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    grosor = 2

    ancho = cv2.getTextSize(
        texto,
        fuente,
        escala,
        grosor
    )[0][0]

    x = (frame.shape[1] - ancho) // 2

    cv2.putText(
        frame,
        texto,
        (x, y),
        fuente,
        escala,
        NEGRO,
        grosor + 4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        texto,
        (x, y),
        fuente,
        escala,
        color,
        grosor,
        cv2.LINE_AA
    )


# Dibuja los puntos y conexiones relevantes
def dibujar_pose(frame, landmarks):
    height, width, _ = frame.shape

    conexiones = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28)
    ]

    for inicio, fin in conexiones:
        p1 = landmarks[inicio]
        p2 = landmarks[fin]

        if (
            p1.visibility < 0.35
            or p2.visibility < 0.35
        ):
            continue

        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            BLANCO,
            2,
            cv2.LINE_AA
        )

    puntos = [
        11, 12,
        13, 14,
        15, 16,
        23, 24,
        25, 26,
        27, 28
    ]

    for indice in puntos:
        landmark = landmarks[indice]

        if landmark.visibility < 0.35:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            6,
            VERDE,
            -1,
            cv2.LINE_AA
        )


# Comprueba si la pierna seleccionada tiene suficiente visibilidad
def pierna_visible(visibilidad):
    return visibilidad >= VISIBILIDAD_MINIMA


# Comprueba si el angulo de rodilla corresponde a una posicion sentada
def posicion_sentada(angulo):
    return angulo < ANGULO_SENTADO_MAX


# Comprueba si la pierna esta suficientemente extendida
def posicion_de_pie(angulo):
    return angulo > ANGULO_DE_PIE_MIN


inicio_programa = time.monotonic()

print("Chair Stand Test")
print("Colocate de perfil respecto a la camara.")
print("Pulsa ESC para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    ahora = time.monotonic()
    tiempo = ahora - inicio_programa

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    resultado = landmarker.detect_for_video(
        mp_image,
        int(tiempo * 1000)
    )

    persona_detectada = False
    pierna_detectada = False
    sentado_detectado = False
    lado_visible = None
    visibilidad = 0
    angulo_rodilla = None

    if resultado.pose_landmarks:
        persona_detectada = True
        landmarks = resultado.pose_landmarks[0]

        dibujar_pose(frame, landmarks)

        lado_visible, visibilidad = obtener_lado_visible(
            landmarks
        )

        pierna_detectada = pierna_visible(
            visibilidad
        )

        cadera, rodilla, tobillo = obtener_pierna(
            landmarks,
            lado_visible
        )

        angulo_rodilla = calcular_angulo(
            cadera,
            rodilla,
            tobillo
        )

        cadera_y = cadera.y

        sentado_detectado = posicion_sentada(
            angulo_rodilla
        )

        de_pie_detectado = posicion_de_pie(
            angulo_rodilla
        )

        if fase == PREPARACION:
            if pierna_detectada and sentado_detectado:
                if tiempo_inicio_estable is None:
                    tiempo_inicio_estable = ahora
                    cadera_sentado = cadera_y

                cadera_sentado = (
                    0.9 * cadera_sentado
                    + 0.1 * cadera_y
                )

                tiempo_estable = (
                    ahora - tiempo_inicio_estable
                )

                if tiempo_estable >= TIEMPO_ESTABLE_INICIAL:
                    fase = LISTO
                    estado = SENTADO
                    tiempo_mensaje_listo = ahora
                    cadera_anterior = cadera_y

            else:
                tiempo_inicio_estable = None
                cadera_sentado = None

        elif fase == LISTO:
            if ahora - tiempo_mensaje_listo >= 2.5:
                fase = PRUEBA
                estado = SENTADO
                cadera_anterior = cadera_y

        elif fase == PRUEBA and not prueba_completada:
            diferencia_cadera = (
                cadera_sentado - cadera_y
            )

            if cadera_anterior is None:
                movimiento_vertical = 0
            else:
                movimiento_vertical = (
                    cadera_anterior - cadera_y
                )

            if estado == SENTADO:
                if (
                    pierna_detectada
                    and diferencia_cadera > UMBRAL_INICIO_SUBIDA
                    and movimiento_vertical > 0
                    and not sentado_detectado
                ):
                    estado = SUBIENDO
                    tiempo_inicio_de_pie = None

            elif estado == SUBIENDO:
                if (
                    abs(cadera_y - cadera_sentado)
                    < UMBRAL_SENTADO
                    and sentado_detectado
                ):
                    estado = SENTADO
                    tiempo_inicio_de_pie = None

                elif (
                    diferencia_cadera > UMBRAL_DE_PIE
                    and de_pie_detectado
                ):
                    if tiempo_inicio_de_pie is None:
                        tiempo_inicio_de_pie = ahora

                    elif (
                        ahora - tiempo_inicio_de_pie
                        >= TIEMPO_CONFIRMAR_DE_PIE
                    ):
                        estado = DE_PIE
                        tiempo_inicio_sentado_final = None

                else:
                    tiempo_inicio_de_pie = None

            elif estado == DE_PIE:
                if movimiento_vertical < -0.002:
                    estado = BAJANDO
                    tiempo_inicio_sentado_final = None

            elif estado == BAJANDO:
                cerca_del_asiento = (
                    abs(cadera_y - cadera_sentado)
                    < UMBRAL_SENTADO
                )

                if (
                    cerca_del_asiento
                    and sentado_detectado
                ):
                    if tiempo_inicio_sentado_final is None:
                        tiempo_inicio_sentado_final = ahora

                    elif (
                        ahora - tiempo_inicio_sentado_final
                        >= TIEMPO_CONFIRMAR_SENTADO
                    ):
                        estado = COMPLETADO
                        prueba_completada = True

                else:
                    tiempo_inicio_sentado_final = None

            cadera_anterior = cadera_y

    if fase == PREPARACION:
        texto_centrado(
            frame,
            "PREPARATE",
            70,
            1.4,
            AMARILLO
        )

        if not persona_detectada:
            texto_centrado(
                frame,
                "Colocate delante de la camara",
                120,
                0.7,
                GRANATE
            )

        elif not pierna_detectada:
            texto_centrado(
                frame,
                "Colocate de perfil y muestra bien una pierna",
                120,
                0.55,
                GRANATE
            )

        elif not sentado_detectado:
            texto_centrado(
                frame,
                "Sientate correctamente",
                120,
                0.7,
                GRANATE
            )

        else:
            texto_centrado(
                frame,
                "Posicion sentada detectada",
                120,
                0.7,
                VERDE
            )

            texto_centrado(
                frame,
                "Manten la posicion",
                160,
                0.65,
                AZUL_GRISACEO
            )

            if tiempo_inicio_estable is not None:
                restante = max(
                    0,
                    TIEMPO_ESTABLE_INICIAL
                    - (
                        time.monotonic()
                        - tiempo_inicio_estable
                    )
                )

                texto_centrado(
                    frame,
                    f"{restante:.1f}",
                    210,
                    0.9,
                    BLANCO
                )

    elif fase == LISTO:
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], frame.shape[0]),
            NEGRO,
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0
        )

        texto_centrado(
            frame,
            "FASE DE PRUEBA",
            frame.shape[0] // 2 - 40,
            1.5,
            AMARILLO
        )

        texto_centrado(
            frame,
            "ESTA REPETICION NO CUENTA",
            frame.shape[0] // 2 + 30,
            0.8,
            AZUL_GRISACEO
        )

        texto_centrado(
            frame,
            "ESPERA...",
            frame.shape[0] // 2 + 90,
            0.7,
            BLANCO
        )

    elif fase == PRUEBA and not prueba_completada:
        texto_centrado(
            frame,
            "REPETICION DE PRUEBA",
            60,
            1.0,
            AMARILLO
        )

        if estado == SENTADO:
            mensaje = "CUANDO QUIERAS, LEVANTATE"
            color_mensaje = VERDE

        elif estado == SUBIENDO:
            mensaje = "LEVANTANDOTE..."
            color_mensaje = AZUL_GRISACEO

        elif estado == DE_PIE:
            mensaje = "DE PIE - AHORA SIENTATE"
            color_mensaje = VERDE

        elif estado == BAJANDO:
            mensaje = "SENTANDOTE..."
            color_mensaje = AZUL_GRISACEO

        else:
            mensaje = estado
            color_mensaje = BLANCO

        texto_centrado(
            frame,
            mensaje,
            110,
            0.7,
            color_mensaje
        )

        cv2.putText(
            frame,
            f"Estado: {estado}",
            (20, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            BLANCO,
            2,
            cv2.LINE_AA
        )

    if persona_detectada and angulo_rodilla is not None:
        cv2.putText(
            frame,
            f"Lado: {lado_visible.lower()}",
            (20, frame.shape[0] - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            GRIS,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Rodilla: {angulo_rodilla:.0f} grados",
            (20, frame.shape[0] - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            GRIS,
            2,
            cv2.LINE_AA
        )

    if prueba_completada:
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], frame.shape[0]),
            NEGRO,
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0
        )

        texto_centrado(
            frame,
            "PRUEBA CORRECTA",
            frame.shape[0] // 2 - 60,
            1.4,
            VERDE
        )

        texto_centrado(
            frame,
            "PREPARADO?",
            frame.shape[0] // 2 + 10,
            1.2,
            AMARILLO
        )

        texto_centrado(
            frame,
            "VA A COMENZAR EL TEST",
            frame.shape[0] // 2 + 80,
            0.8,
            AZUL_GRISACEO
        )

    cv2.imshow(
        "30-second Chair Stand Test",
        frame
    )

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()