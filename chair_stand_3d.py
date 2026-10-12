import math
import time
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/pose_landmarker_full.task"

DURACION = 30
PREPARACION_SEG = 2
CONFIRMAR_PIE = 0.5
CONFIRMAR_SENTADO = 0.5

ANGULO_SENTADO = 140
ANGULO_PIE = 155

MIN_SUBIDA = 0.035
ALTURA_PIE = 0.09
TOLERANCIA_ASIENTO = 0.055

GRANATE = (90, 90, 190)
VERDE = (100, 190, 100)
AMARILLO = (140, 205, 220)
BLANCO = (235, 235, 235)
NEGRO = (20, 20, 20)

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

    if denominador < 1e-8:
        return None

    coseno = np.dot(v1, v2) / denominador
    coseno = np.clip(coseno, -1, 1)

    return math.degrees(math.acos(coseno))


def dibujar_texto(frame, mensaje, y, escala=0.8, color=BLANCO):
    alto, ancho = frame.shape[:2]

    fuente = cv2.FONT_HERSHEY_SIMPLEX

    ancho_texto = cv2.getTextSize(
        mensaje, fuente, escala, 2
    )[0][0]

    if ancho_texto > ancho - 20:
        escala *= (ancho - 20) / ancho_texto

    ancho_texto = cv2.getTextSize(
        mensaje, fuente, escala, 2
    )[0][0]

    x = max(5, (ancho - ancho_texto) // 2)

    cv2.putText(
        frame, mensaje, (x, y),
        fuente, escala, NEGRO, 6, cv2.LINE_AA
    )

    cv2.putText(
        frame, mensaje, (x, y),
        fuente, escala, color, 2, cv2.LINE_AA
    )


def detectar_pierna(world, imagen):
    opciones = []

    for indices in [(23, 25, 27), (24, 26, 28)]:
        visibilidad = min(
            imagen[i].visibility for i in indices
        )

        if visibilidad < 0.45:
            continue

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

    if not opciones:
        return None, None

    _, angulo, cadera = max(opciones)

    return angulo, cadera


def comprobar_brazo(world, imagen, lado):
    if lado == "izquierdo":
        hombro, codo, muneca = 11, 13, 15
        hombro_opuesto = 12
    else:
        hombro, codo, muneca = 12, 14, 16
        hombro_opuesto = 11

    visibilidad = min(
        imagen[hombro].visibility,
        imagen[codo].visibility,
        imagen[muneca].visibility
    )

    if visibilidad < 0.45:
        return None

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

    distancia = min(
        np.linalg.norm(p_muneca - p_hombro),
        np.linalg.norm(p_muneca - p_centro)
    )

    return flexion < 150 and distancia < 2 * escala


def brazos_cruzados(world, imagen):
    izquierdo = comprobar_brazo(
        world, imagen, "izquierdo"
    )

    derecho = comprobar_brazo(
        world, imagen, "derecho"
    )

    visibles = [
        valor for valor in [izquierdo, derecho]
        if valor is not None
    ]

    if not visibles:
        return None

    return all(visibles)


def dibujar_pose(frame, landmarks):
    alto, ancho = frame.shape[:2]

    for i, j in CONEXIONES:
        if min(
            landmarks[i].visibility,
            landmarks[j].visibility
        ) < 0.35:
            continue

        a = (
            int(landmarks[i].x * ancho),
            int(landmarks[i].y * alto)
        )

        b = (
            int(landmarks[j].x * ancho),
            int(landmarks[j].y * alto)
        )

        cv2.line(
            frame, a, b, VERDE, 2, cv2.LINE_AA
        )

        cv2.circle(frame, a, 4, BLANCO, -1)
        cv2.circle(frame, b, 4, BLANCO, -1)


class ChairStand3D:

    def __init__(self):
        self.fase = "PREPARACION"
        self.estado = "SENTADO"

        self.referencia = None
        self.estable_desde = None
        self.fase_desde = None
        self.test_desde = None

        self.confirmacion = None
        self.cadera_anterior = None

        self.repeticiones = 0
        self.errores = Counter()

        self.aviso = ""
        self.aviso_desde = 0

        self.brazos_mal_desde = None
        self.error_brazos = False
        self.ha_estado_de_pie = False

    def mostrar_aviso(self, mensaje):
        self.aviso = mensaje
        self.aviso_desde = time.monotonic()

    def reiniciar_movimiento(self):
        self.estado = "SENTADO"
        self.confirmacion = None
        self.cadera_anterior = None
        self.brazos_mal_desde = None
        self.error_brazos = False
        self.ha_estado_de_pie = False

    def actualizar_tiempo(self, ahora):
        if self.fase == "LISTO":
            if ahora - self.fase_desde >= 2.5:
                self.fase = "PRUEBA"
                self.reiniciar_movimiento()

        elif self.fase == "TRANSICION":
            if ahora - self.fase_desde >= 3:
                self.fase = "TEST"
                self.test_desde = ahora
                self.repeticiones = 0
                self.errores.clear()
                self.reiniciar_movimiento()

                print("TEST INICIADO")

        elif self.fase == "TEST":
            if ahora - self.test_desde >= DURACION:
                self.fase = "FINALIZADO"

                print("TEST FINALIZADO")
                print("Repeticiones:", self.repeticiones)
                print("Errores:", dict(self.errores))

    def evaluar_brazos(self, correcto, ahora):
        if self.fase != "TEST":
            return

        if correcto is False:
            if self.brazos_mal_desde is None:
                self.brazos_mal_desde = ahora

            elif ahora - self.brazos_mal_desde >= 0.5:
                self.error_brazos = True

        else:
            self.brazos_mal_desde = None

    def finalizar_repeticion(self, ahora):
        if self.fase == "PRUEBA":
            self.fase = "TRANSICION"
            self.fase_desde = ahora

            print("REPETICION DE PRUEBA CORRECTA")

        elif self.fase == "TEST":
            if self.error_brazos:
                self.errores["brazos"] += 1

                self.mostrar_aviso(
                    "MANTEN LOS BRAZOS CRUZADOS"
                )

                print("REPETICION INVALIDA: BRAZOS")

            else:
                self.repeticiones += 1

                print(
                    "REPETICION CORRECTA:",
                    self.repeticiones
                )

        self.reiniciar_movimiento()

    def actualizar_pose(self, ahora, rodilla, cadera, brazos):
        if rodilla is None or cadera is None:
            self.cadera_anterior = None
            self.confirmacion = None
            return

        sentado = rodilla < ANGULO_SENTADO
        de_pie = rodilla > ANGULO_PIE

        if self.fase == "PREPARACION":
            if sentado:
                if self.estable_desde is None:
                    self.estable_desde = ahora
                    self.referencia = cadera

                self.referencia = (
                    0.92 * self.referencia +
                    0.08 * cadera
                )

                if ahora - self.estable_desde >= PREPARACION_SEG:
                    self.fase = "LISTO"
                    self.fase_desde = ahora
                    self.reiniciar_movimiento()

            else:
                self.estable_desde = None
                self.referencia = None

            return

        if self.fase not in ("PRUEBA", "TEST"):
            return

        subida = self.referencia - cadera

        velocidad = (
            0 if self.cadera_anterior is None
            else self.cadera_anterior - cadera
        )

        cerca_silla = (
            abs(self.referencia - cadera)
            < TOLERANCIA_ASIENTO
        )

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
            self.evaluar_brazos(brazos, ahora)

            if cerca_silla and sentado:
                if self.fase == "TEST":
                    self.errores["extension"] += 1

                    self.mostrar_aviso(
                        "LEVANTATE COMPLETAMENTE"
                    )

                self.reiniciar_movimiento()

            elif subida > ALTURA_PIE and de_pie:
                if self.confirmacion is None:
                    self.confirmacion = ahora

                elif ahora - self.confirmacion >= CONFIRMAR_PIE:
                    self.estado = "DE_PIE"
                    self.ha_estado_de_pie = True
                    self.confirmacion = None

            else:
                self.confirmacion = None

                if velocidad < -0.004:
                    self.estado = "BAJANDO"

        elif self.estado == "DE_PIE":
            self.evaluar_brazos(brazos, ahora)

            if velocidad < -0.002:
                self.estado = "BAJANDO"
                self.confirmacion = None

        elif self.estado == "BAJANDO":
            self.evaluar_brazos(brazos, ahora)

            if cerca_silla and sentado:
                if self.confirmacion is None:
                    self.confirmacion = ahora

                elif (
                    ahora - self.confirmacion
                    >= CONFIRMAR_SENTADO
                ):
                    if self.ha_estado_de_pie:
                        self.finalizar_repeticion(ahora)

                    else:
                        if self.fase == "TEST":
                            self.errores["extension"] += 1

                            self.mostrar_aviso(
                                "LEVANTATE COMPLETAMENTE"
                            )

                        self.reiniciar_movimiento()

            else:
                self.confirmacion = None

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

        if (
            self.fase == "TEST"
            and self.aviso
            and ahora - self.aviso_desde < 1.8
        ):
            dibujar_texto(
                frame, self.aviso,
                alto - 80, 0.65, GRANATE
            )

        if (
            self.fase in ("PRUEBA", "TEST")
            and not detectado
        ):
            dibujar_texto(
                frame, "POSE NO DETECTADA",
                alto - 20, 0.55, GRANATE
            )


def main():
    opciones = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    detector = vision.PoseLandmarker.create_from_options(
        opciones
    )

    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        detector.close()
        raise RuntimeError("No se puede abrir la webcam")

    plt.ion()

    figura = plt.figure(figsize=(6, 6))
    eje = figura.add_subplot(111, projection="3d")

    figura.canvas.manager.set_window_title(
        "Chair Stand - Esqueleto 3D"
    )

    eje.set_xlabel("X")
    eje.set_ylabel("Profundidad Z")
    eje.set_zlabel("Altura")

    eje.set_xlim(-1, 1)
    eje.set_ylim(-1, 1)
    eje.set_zlim(-1, 1)

    eje.view_init(elev=15, azim=-70)

    lineas = [
        eje.plot(
            [], [], [],
            color="royalblue",
            linewidth=2.5
        )[0]
        for _ in CONEXIONES
    ]

    puntos = eje.scatter(
        [], [], [],
        color="darkred",
        s=20
    )

    test = ChairStand3D()

    inicio = time.monotonic()
    ultimo_timestamp = -1
    numero_frame = 0

    print("CHAIR STAND TEST 3D INICIADO")
    print("Pulsa ESC para salir")

    try:
        while True:
            correcto, frame = camara.read()

            if not correcto:
                break

            ahora = time.monotonic()

            test.actualizar_tiempo(ahora)

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            imagen_mp = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = max(
                ultimo_timestamp + 1,
                int((ahora - inicio) * 1000)
            )

            ultimo_timestamp = timestamp

            resultado = detector.detect_for_video(
                imagen_mp,
                timestamp
            )

            detectado = bool(
                resultado.pose_landmarks
                and resultado.pose_world_landmarks
            )

            if detectado:
                imagen = resultado.pose_landmarks[0]
                world = resultado.pose_world_landmarks[0]

                dibujar_pose(frame, imagen)

                rodilla, cadera = detectar_pierna(
                    world, imagen
                )

                brazos = (
                    brazos_cruzados(world, imagen)
                    if test.fase == "TEST"
                    else None
                )

                test.actualizar_pose(
                    ahora, rodilla, cadera, brazos
                )

                if numero_frame % 3 == 0:
                    coordenadas = np.array([
                        [p.x, p.y, p.z]
                        for p in world
                    ])

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

                        linea.set_3d_properties([
                            -coordenadas[i, 1],
                            -coordenadas[j, 1]
                        ])

                    puntos._offsets3d = (
                        coordenadas[:, 0],
                        coordenadas[:, 2],
                        -coordenadas[:, 1]
                    )

            else:
                test.cadera_anterior = None
                test.confirmacion = None

            test.dibujar_interfaz(
                frame, ahora, detectado
            )

            cv2.imshow(
                "Chair Stand Test 3D - Webcam",
                frame
            )

            numero_frame += 1

            if numero_frame % 3 == 0:
                figura.canvas.draw_idle()
                plt.pause(0.001)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                break

            if not plt.fignum_exists(figura.number):
                break

    finally:
        camara.release()
        detector.close()
        cv2.destroyAllWindows()
        plt.close("all")


if __name__ == "__main__":
    main()