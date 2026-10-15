import math
import time
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Modelo de MediaPipe utilizado para detectar la postura corporal.
MODEL_PATH = "models/pose_landmarker_full.task"

# Duración del test y tiempos de confirmación de las posturas.
DURACION = 30
PREPARACION_SEG = 2
CONFIRMAR_PIE = 0.5
CONFIRMAR_SENTADO = 0.5

# Ángulos de rodilla que permiten identificar las posturas.
ANGULO_SENTADO = 140
ANGULO_PIE = 155


# Umbrales para detectar el movimiento vertical de la cadera.
MIN_SUBIDA = 0.035
ALTURA_PIE = 0.09
TOLERANCIA_ASIENTO = 0.055

GRANATE = (90, 90, 190)
VERDE = (100, 190, 100)
AMARILLO = (140, 205, 220)
BLANCO = (235, 235, 235)
NEGRO = (20, 20, 20)


# Parejas de puntos
CONEXIONES = [
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

# Calcula el ángulo formado por tres articulaciones
def angulo_3d(a, b, c):
    v1 = np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ])

    v2 = np.array([
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ])

    denominador = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Evita dividir entre cero
    if denominador < 1e-8:
        return None

    coseno = np.dot(v1, v2) / denominador
    coseno = np.clip(coseno, -1, 1)

    return math.degrees(math.acos(coseno))


def dibujar_texto(frame, mensaje, y, escala=0.8, color=BLANCO):
    alto, ancho = frame.shape[:2]

    fuente = cv2.FONT_HERSHEY_SIMPLEX

    # Calcula cuánto espacio ocupa el texto
    ancho_texto = cv2.getTextSize(
        mensaje, fuente, escala, 2
    )[0][0]

     # Reduce el tamaño si el texto supera el ancho de la imagen
    if ancho_texto > ancho - 20:
        escala *= (ancho - 20) / ancho_texto

    ancho_texto = cv2.getTextSize(
        mensaje, fuente, escala, 2
    )[0][0]

    #Centra el mensaje
    x = max(5, (ancho - ancho_texto) // 2)

    # Dibuja contorno negro
    cv2.putText(
        frame, mensaje, (x, y),
        fuente, escala, NEGRO, 6, cv2.LINE_AA
    )

    # Dibuja el texto con su color
    cv2.putText(
        frame, mensaje, (x, y),
        fuente, escala, color, 2, cv2.LINE_AA
    )

# Selecciona la pierna más visible y calcula su ángulo de rodilla
def detectar_pierna(world, imagen):
    opciones = []

    # Pierna izquierda: cadera 23, rodilla 25 y tobillo 27
    # Pierna derecha: cadera 24, rodilla 26 y tobillo 28
    for indices in [(23, 25, 27), (24, 26, 28)]:
        visibilidad = min(
            imagen[i].visibility for i in indices
        )

        # Descarta la pierna si alguno de sus puntos es poco visible
        if visibilidad < 0.45:
            continue

        # Calcula el ángulo tridimensional de la rodilla
        angulo = angulo_3d(
            world[indices[0]],
            world[indices[1]],
            world[indices[2]]
        )

        if angulo is not None:
            opciones.append((
                visibilidad,
                angulo,
                imagen[indices[0]].y
            ))

    # Si ninguna pierna es suficientemente visible, no se mide
    if not opciones:
        return None, None

    # Selecciona la pierna con mayor visibilidad
    _, angulo, cadera = max(opciones)

    return angulo, cadera

 # Comprueba si un brazo está flexionado y próximo al pecho
def comprobar_brazo(world, imagen, lado):

    # Selecciona las articulaciones de los brazos
    if lado == "izquierdo":
        hombro, codo, muneca = 11, 13, 15
        hombro_opuesto = 12
    else:
        hombro, codo, muneca = 12, 14, 16
        hombro_opuesto = 11

    # Comprueba la visibilidad del hombro, codo y muñeca
    visibilidad = min(
        imagen[hombro].visibility,
        imagen[codo].visibility,
        imagen[muneca].visibility
    )

    # Si el brazo no es suficientemente visible, no se mide
    if visibilidad < 0.45:
        return None

    # Calcula el ángulo del codo 
    flexion = angulo_3d(
        world[hombro],
        world[codo],
        world[muneca]
    )

    if flexion is None:
        return None

    p_muneca = np.array([
        world[muneca].x,
        world[muneca].y,
        world[muneca].z
    ])

    p_hombro = np.array([
        world[hombro_opuesto].x,
        world[hombro_opuesto].y,
        world[hombro_opuesto].z
    ])

    p_centro = np.array([
        (world[11].x + world[12].x) / 2,
        (world[11].y + world[12].y) / 2,
        (world[11].z + world[12].z) / 2
    ])

    ancho_hombros = np.linalg.norm(
        np.array([
            world[11].x - world[12].x,
            world[11].y - world[12].y,
            world[11].z - world[12].z
        ])
    )

    escala = max(ancho_hombros, 0.18)

    # Obtiene la menor distancia de la muñeca al hombro contrario
    # o al centro de los hombros
    distancia = min(
        np.linalg.norm(p_muneca - p_hombro),
        np.linalg.norm(p_muneca - p_centro)
    )

    # Considera correcta la posición si el codo está flexionado
    # y la muñeca permanece suficientemente cerca del pecho
    return flexion < 150 and distancia < 2 * escala

# Comprueba que los brazos esten cruzados
def brazos_cruzados(world, imagen):
    izquierdo = comprobar_brazo(
        world, imagen, "izquierdo"
    )

    derecho = comprobar_brazo(
        world, imagen, "derecho"
    )

    # Descarta los brazos que no se han podido evaluar
    visibles = [
        valor for valor in [izquierdo, derecho]
        if valor is not None
    ]

    if not visibles:
        return None

    return all(visibles)

# Dibuja el esqueleto sobre la imagen original
def dibujar_pose(frame, landmarks):
    alto, ancho = frame.shape[:2]

    for i, j in CONEXIONES:

        # Evita dibujar conexiones cuyos puntos sean poco visibles
        if min(
            landmarks[i].visibility,
            landmarks[j].visibility
        ) < 0.35:
            continue

        # Convierte las coordenadas normalizadas a píxeles
        a = (
            int(landmarks[i].x * ancho),
            int(landmarks[i].y * alto)
        )

        b = (
            int(landmarks[j].x * ancho),
            int(landmarks[j].y * alto)
        )

        # Dibuja la línea que une las dos articulaciones
        cv2.line(
            frame, a, b, VERDE, 2, cv2.LINE_AA
        )

        # Marca las articulaciones mediante círculos
        cv2.circle(frame, a, 4, BLANCO, -1)
        cv2.circle(frame, b, 4, BLANCO, -1)


class ChairStand3D:

    def __init__(self):
        self.fase = "PREPARACION"
        self.estado = "SENTADO"

        # Referencia inicial de la cadera y tiempos de cada fase
        self.referencia = None
        self.estable_desde = None
        self.fase_desde = None
        self.test_desde = None

        # Variables utilizadas para confirmar posturas y detectar movimiento
        self.confirmacion = None
        self.cadera_anterior = None

        # Contadores
        self.repeticiones = 0
        self.errores = Counter()

        # Mensaje temporal que se muestra cuando ocurre un error
        self.aviso = ""
        self.aviso_desde = 0

        # Variables para comprobar los brazos
        self.brazos_mal_desde = None
        self.error_brazos = False
        self.ha_estado_de_pie = False

    # Guarda el mensaje y el momento en el que debe empezar a mostrarse
    def mostrar_aviso(self, mensaje):
        self.aviso = mensaje
        self.aviso_desde = time.monotonic()

    # Reinicia las variables de movimiento para comenzar otra repetición
    def reiniciar_movimiento(self):
        self.estado = "SENTADO"
        self.confirmacion = None
        self.cadera_anterior = None
        self.brazos_mal_desde = None
        self.error_brazos = False
        self.ha_estado_de_pie = False

    # Controla los cambios de fase, permite que el test termine aunque se pierda la postura
    def actualizar_tiempo(self, ahora):

        # Tras 2,5 segundos comienza la repetición de prueba
        if self.fase == "LISTO":
            if ahora - self.fase_desde >= 2.5:
                self.fase = "PRUEBA"
                self.reiniciar_movimiento()

        # Tras 3 segundos comienza el test real
        elif self.fase == "TRANSICION":
            if ahora - self.fase_desde >= 3:
                self.fase = "TEST"
                self.test_desde = ahora

                # Reinicia los contadores
                self.repeticiones = 0
                self.errores.clear()
                self.reiniciar_movimiento()

                print("TEST INICIADO")

        elif self.fase == "TEST":

            # Finaliza al alcanzar los 30 segundos
            if ahora - self.test_desde >= DURACION:
                self.fase = "FINALIZADO"

                print("TEST FINALIZADO")
                print("Repeticiones:", self.repeticiones)
                print("Errores:", dict(self.errores))

    # Comprueba si los brazos
    def evaluar_brazos(self, correcto, ahora):
        if self.fase != "TEST":
            return

        if correcto is False:
            if self.brazos_mal_desde is None:
                self.brazos_mal_desde = ahora

            elif ahora - self.brazos_mal_desde >= 0.5:
                self.error_brazos = True

        else:

             # Reinicia el tiempo de error si la postura deja de ser incorrecta
            self.brazos_mal_desde = None

    # Se ejecuta cuando se ha completado la secuencia
    # de levantarse y volver a sentarse.
    def finalizar_repeticion(self, ahora):
        if self.fase == "PRUEBA":
            # La repetición de prueba no se añade al contador
            self.fase = "TRANSICION"
            self.fase_desde = ahora

            print("REPETICION DE PRUEBA CORRECTA")

        elif self.fase == "TEST":
            if self.error_brazos:

                # Si los brazos han estado mal colocados, no suma
                self.errores["brazos"] += 1

                self.mostrar_aviso(
                    "MANTEN LOS BRAZOS CRUZADOS"
                )

                print("REPETICION INVALIDA: BRAZOS")

            else:

                # Si no se ha detectado el error, suma una repetición
                self.repeticiones += 1

                print(
                    "REPETICION CORRECTA:",
                    self.repeticiones
                )

        self.reiniciar_movimiento()

    def actualizar_pose(self, ahora, rodilla, cadera, brazos):

        # Si faltan datos de rodilla o cadera, no actualiza el movimiento
        if rodilla is None or cadera is None:
            self.cadera_anterior = None
            self.confirmacion = None
            return

        # Clasifica la postura según el ángulo de la rodilla
        sentado = rodilla < ANGULO_SENTADO
        de_pie = rodilla > ANGULO_PIE

        if self.fase == "PREPARACION":

            # Calibra la posición inicial mientras la persona está sentada
            if sentado:
                if self.estable_desde is None:
                    self.estable_desde = ahora
                    self.referencia = cadera

                # Suaviza la altura de referencia
                self.referencia = (
                    0.92 * self.referencia +
                    0.08 * cadera
                )

                # Tras 2 segundos sentado, pasa a la fase de prueba
                if ahora - self.estable_desde >= PREPARACION_SEG:
                    self.fase = "LISTO"
                    self.fase_desde = ahora
                    self.reiniciar_movimiento()

            else:

                # Si deja de estar sentado, reinicia la calibración
                self.estable_desde = None
                self.referencia = None

            return

        # Solo analiza repeticiones durante la prueba y el test
        if self.fase not in ("PRUEBA", "TEST"):
            return

        subida = self.referencia - cadera

        # Positivo: la cadera sube. Negativo: la cadera baja
        velocidad = (
            0 if self.cadera_anterior is None
            else self.cadera_anterior - cadera
        )

        # Comprueba si la cadera ha regresado cerca de la altura inicial
        cerca_silla = (
            abs(self.referencia - cadera)
            < TOLERANCIA_ASIENTO
        )

        # Detecta el inicio de una subida.
        # La cadera debe elevarse y la rodilla dejar de estar flexionada.
        if self.estado == "SENTADO":
            if (
                subida > MIN_SUBIDA
                and velocidad > 0
                and not sentado
            ):
                self.estado = "SUBIENDO"
                self.confirmacion = None
                self.error_brazos = False
                self.ha_estado_de_pie = False

        elif self.estado == "SUBIENDO":

            # Comprueba la posición de los brazos
            self.evaluar_brazos(brazos, ahora)

            if cerca_silla and sentado:

                # Si vuelve a sentarse sin completar la subida
                # registra un error de extensión
                if self.fase == "TEST":
                    self.errores["extension"] += 1

                    self.mostrar_aviso(
                        "LEVANTATE COMPLETAMENTE"
                    )

                self.reiniciar_movimiento()

            elif subida > ALTURA_PIE and de_pie:

                # Comprueba que alcanza suficiente altura
                # y que la rodilla está extendida
                if self.confirmacion is None:
                    self.confirmacion = ahora

                elif ahora - self.confirmacion >= CONFIRMAR_PIE:

                    # Confirma la postura de pie
                    self.estado = "DE_PIE"
                    self.ha_estado_de_pie = True
                    self.confirmacion = None

            else:
                self.confirmacion = None

                # Si empieza a descender, cambia al estado BAJANDO
                if velocidad < -0.004:
                    self.estado = "BAJANDO"

        elif self.estado == "DE_PIE":

            # Continúa comprobando los brazos mientras está de pie
            self.evaluar_brazos(brazos, ahora)

            # Detecta el inicio del descenso
            if velocidad < -0.002:
                self.estado = "BAJANDO"
                self.confirmacion = None

        elif self.estado == "BAJANDO":

            # Comprueba los brazos durante el descenso
            self.evaluar_brazos(brazos, ahora)

            if cerca_silla and sentado:
                if self.confirmacion is None:
                    self.confirmacion = ahora

                elif (
                    ahora - self.confirmacion
                    >= CONFIRMAR_SENTADO
                ):
                    if self.ha_estado_de_pie:

                        # Completa la repetición si antes estuvo de pie
                        self.finalizar_repeticion(ahora)

                    else:

                        # Si nunca confirmó estar de pie, la repetición
                        # se considera incompleta
                        if self.fase == "TEST":
                            self.errores["extension"] += 1

                            self.mostrar_aviso(
                                "LEVANTATE COMPLETAMENTE"
                            )

                        self.reiniciar_movimiento()

            else:
                self.confirmacion = None

                # Detecta si vuelve a subir antes de llegar al asiento
                if (
                    velocidad > 0.006
                    and subida > TOLERANCIA_ASIENTO
                ):
                    if self.fase == "TEST":
                        self.errores["silla"] += 1

                        self.mostrar_aviso(
                            "NO HAS LLEGADO A SENTARTE"
                        )

                    self.estado = "SUBIENDO"
                    self.ha_estado_de_pie = False
                    self.confirmacion = None
                    self.error_brazos = False

        self.cadera_anterior = cadera

    # Muestra las indicaciones de cada fase del test
    def dibujar_interfaz(self, frame, ahora, detectado):
        alto = frame.shape[0]

        if self.fase == "PREPARACION":
            dibujar_texto(
                frame, "PREPARATE", 65, 1.1, AMARILLO
            )

            dibujar_texto(
                frame,
                "SIENTATE Y MANTEN LA POSICION"
                if detectado else "COLOCATE DE PERFIL",
                115, 0.7
            )

        elif self.fase == "LISTO":
            dibujar_texto(
                frame, "FASE DE PRUEBA",
                alto // 2 - 35, 1.1, AMARILLO
            )

            dibujar_texto(
                frame, "LA REPETICION NO CUENTA",
                alto // 2 + 25, 0.75
            )

        elif self.fase == "PRUEBA":
            dibujar_texto(
                frame, "REPETICION DE PRUEBA",
                65, 0.9, AMARILLO
            )

            dibujar_texto(
                frame, self.estado,
                alto - 45, 0.8, VERDE
            )

        elif self.fase == "TRANSICION":
            dibujar_texto(
                frame, "PRUEBA CORRECTA",
                alto // 2 - 45, 1.1, VERDE
            )

            dibujar_texto(
                frame, "EL TEST COMIENZA EN BREVE",
                alto // 2 + 25, 0.75, AMARILLO
            )

        elif self.fase == "TEST":
        # Durante el test se muestra el contador

            dibujar_texto(
                frame, "CHAIR STAND TEST 3D",
                55, 0.85, AMARILLO
            )

            dibujar_texto(
                frame, str(self.repeticiones),
                alto // 2, 3.0
            )

            dibujar_texto(
                frame, "REPETICIONES CORRECTAS",
                alto // 2 + 60, 0.75, VERDE
            )

        elif self.fase == "FINALIZADO":
        # Al terminar muestra las repeticiones y el total de errores

            dibujar_texto(
                frame, "TEST FINALIZADO",
                alto // 2 - 100, 1.1, AMARILLO
            )

            dibujar_texto(
                frame, str(self.repeticiones),
                alto // 2 - 10, 2.5, VERDE
            )

            dibujar_texto(
                frame, "REPETICIONES CORRECTAS",
                alto // 2 + 35, 0.7
            )

            dibujar_texto(
                frame,
                f"ERRORES: {sum(self.errores.values())}",
                alto // 2 + 90, 0.7, AMARILLO
            )

        # Muestra temporalmente el último aviso
        if (
            self.fase == "TEST"
            and self.aviso
            and ahora - self.aviso_desde < 1.8
        ):
            dibujar_texto(
                frame, self.aviso,
                alto - 80, 0.65, GRANATE
            )

        # Advierte si se pierde la detección de la persona
        if (
            self.fase in ("PRUEBA", "TEST")
            and not detectado
        ):
            dibujar_texto(
                frame, "POSE NO DETECTADA",
                alto - 20, 0.55, GRANATE
            )


def main():

    # Configura MediaPipe para analizar vídeo y detectar una persona
    opciones = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

     # Carga el modelo de estimación de postura
    detector = vision.PoseLandmarker.create_from_options(
        opciones
    )

    # Abre la webcam
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        detector.close()
        raise RuntimeError("No se puede abrir la webcam")

    plt.ion()

    # Crea la ventana donde se representará el esqueleto 3D
    figura = plt.figure(figsize=(6, 6))
    eje = figura.add_subplot(111, projection="3d")

    figura.canvas.manager.set_window_title(
        "Chair Stand - Esqueleto 3D"
    )

    # Configura los ejes de la representación tridimensional
    eje.set_xlabel("X")
    eje.set_ylabel("Profundidad Z")
    eje.set_zlabel("Altura")

    eje.set_xlim(-1, 1)
    eje.set_ylim(-1, 1)
    eje.set_zlim(-1, 1)

# Establece el ángulo inicial desde el que se observa el esqueleto
    eje.view_init(elev=15, azim=-70)

    # Crea las líneas del esqueleto, inicialmente vacías
    lineas = [
        eje.plot(
            [], [], [],
            color="royalblue",
            linewidth=2.5
        )[0]
        for _ in CONEXIONES
    ]

    # Crea los puntos que representan las articulaciones
    puntos = eje.scatter(
        [], [], [],
        color="darkred",
        s=20
    )

    test = ChairStand3D()

    # Variables para controlar los tiempos y los fotogramas
    inicio = time.monotonic()
    ultimo_timestamp = -1
    numero_frame = 0

    print("CHAIR STAND TEST 3D INICIADO")
    print("Pulsa ESC para salir")

    try:

        # Procesa las imágenes de la webcam
        while True:
            correcto, frame = camara.read()

            if not correcto:
                break

            ahora = time.monotonic()

            # Actualiza las fases del test
            test.actualizar_tiempo(ahora)

            # MediaPipe necesita recibir las imágenes en formato RGB
            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            imagen_mp = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

             # Calcula un tiempo en milisegundos
            timestamp = max(
                ultimo_timestamp + 1,
                int((ahora - inicio) * 1000)
            )

            ultimo_timestamp = timestamp

            # Detecta las articulaciones corporales del fotograma
            resultado = detector.detect_for_video(
                imagen_mp,
                timestamp
            )

            # Comprueba que existen puntos corporales
            detectado = bool(
                resultado.pose_landmarks
                and resultado.pose_world_landmarks
            )

            if detectado:

                # Coordenadas normalizadas
                imagen = resultado.pose_landmarks[0]

                # Coordenadas tridimensionales estimadas
                world = resultado.pose_world_landmarks[0]

                dibujar_pose(frame, imagen)

                # Obtiene el ángulo de rodilla y la altura de cadera
                rodilla, cadera = detectar_pierna(
                    world, imagen
                )

                # Los brazos solo se comprueban durante el test real
                brazos = (
                    brazos_cruzados(world, imagen)
                    if test.fase == "TEST"
                    else None
                )

                test.actualizar_pose(
                    ahora, rodilla, cadera, brazos
                )

                # Actualiza el esqueleto 3D cada tres fotogramas
                if numero_frame % 3 == 0:
                    coordenadas = np.array([
                        [p.x, p.y, p.z]
                        for p in world
                    ])

                    # Actualiza cada segmento del esqueleto 3D
                    for linea, (i, j) in zip(
                        lineas, CONEXIONES
                    ):
                        linea.set_data(
                            [
                                coordenadas[i, 0],
                                coordenadas[j, 0]
                            ],
                            [
                                coordenadas[i, 2],
                                coordenadas[j, 2]
                            ]
                        )

                        # Invierte Y para representar la altura hacia arriba
                        linea.set_3d_properties([
                            -coordenadas[i, 1],
                            -coordenadas[j, 1]
                        ])

                    # Actualiza la posición tridimensional de las articulaciones
                    puntos._offsets3d = (
                        coordenadas[:, 0],
                        coordenadas[:, 2],
                        -coordenadas[:, 1]
                    )

            else:

                # Si se pierde la postura, reinicia las mediciones
                test.cadera_anterior = None
                test.confirmacion = None

            # Dibuja el contador, las indicaciones y los avisos
            test.dibujar_interfaz(
                frame, ahora, detectado
            )

            # Muestra la imagen
            cv2.imshow(
                "Chair Stand Test 3D - Webcam",
                frame
            )

            numero_frame += 1

            # Actualiza la ventana tridimensional cada tres fotogramas
            if numero_frame % 3 == 0:
                figura.canvas.draw_idle()
                plt.pause(0.001)

            # Permite salir pulsando ESC
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                break

            if not plt.fignum_exists(figura.number):
                break

    finally:

        # Libera la cámara, el detector y las ventanas al terminar
        camara.release()
        detector.close()
        cv2.destroyAllWindows()
        plt.close("all")


if __name__ == "__main__":
    main()