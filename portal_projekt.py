import cv2
import mediapipe as mp
import numpy as np

mp_kezek = mp.solutions.hands          # a kézfelismerő modul
mp_rajz = mp.solutions.drawing_utils   # segédfüggvény a pontok kirajzolásához

HUVELYK_HEGY = 4    # a 21 landmark közül a hüvelykujj hegye
MUTATO_HEGY = 8     # a mutatóujj hegye
CSIPES_KUSZOB = 0.05  # ennél kisebb távolság = csípés (normalizált egységben)
NEGYSZOG_SZIN = (0, 255, 0)  # BGR: zöld
FEHER = 255  # a maszkban: "bent"

# A választható szűrők: (név, OpenCV színskála)
SZUROK = [
    ("CYBER_NEON", cv2.COLORMAP_PLASMA),
    ("LAVA", cv2.COLORMAP_HOT),
    ("OCEAN", cv2.COLORMAP_OCEAN),
    ("RAINBOW", cv2.COLORMAP_JET),
]


def pixelbe(pont, szelesseg, magassag):
    """Normalizált (0-1) landmarkot pixelkoordinátává alakít, (x, y) tuple-ként."""
    return (int(pont.x * szelesseg), int(pont.y * magassag))


kamera = cv2.VideoCapture(0)  # 0 = alapértelmezett webkamera

# Állapot, aminek képkockáról képkockára meg kell maradnia → a cikluson KÍVÜL
aktualis_szuro = 0       # hányadik szűrő van kiválasztva a listából
elozo_csipes = False     # volt-e csípés az előző képkockán

with mp_kezek.Hands(max_num_hands=2, min_detection_confidence=0.7) as kezek:
    while True:
        sikerult, kep = kamera.read()   # egy képkocka beolvasása
        if not sikerult:
            break

        kep = cv2.flip(kep, 1)  # tükrözés
        rgb = cv2.cvtColor(kep, cv2.COLOR_BGR2RGB)
        eredmeny = kezek.process(rgb)

        magassag, szelesseg, _ = kep.shape

        sarkok = []          # ennek a képkockának a sarokpontjai
        van_csipes = False   # van-e csípés EBBEN a képkockában

        if eredmeny.multi_hand_landmarks:          # találtunk-e kezet?
            for kez in eredmeny.multi_hand_landmarks:
                mp_rajz.draw_landmarks(kep, kez, mp_kezek.HAND_CONNECTIONS)

                huvelyk = kez.landmark[HUVELYK_HEGY]
                mutato = kez.landmark[MUTATO_HEGY]

                tavolsag = np.linalg.norm(np.array([huvelyk.x, huvelyk.y]) - np.array([mutato.x, mutato.y]))
                if tavolsag <= CSIPES_KUSZOB:
                    van_csipes = True

                # Kezenként két sarok: a hüvelykujj és a mutatóujj hegye
                sarkok.append(pixelbe(huvelyk, szelesseg, magassag))
                sarkok.append(pixelbe(mutato, szelesseg, magassag))

        # Csak a csípés kezdetének pillanatában váltunk:
        # most van csípés, de az előző képkockán még nem volt
        if van_csipes and not elozo_csipes:
            aktualis_szuro = (aktualis_szuro + 1) % len(SZUROK)

        elozo_csipes = van_csipes  # megjegyezzük a következő képkockának

        szuro_nev, szinskala = SZUROK[aktualis_szuro]

        # Csak akkor rajzolunk, ha mindkét kéz megvan (2 kéz × 2 pont)
        if len(sarkok) == 4:
            pontok = np.array(sarkok, dtype=np.int32)
            burok = cv2.convexHull(pontok)   # sorba rendezi a sarkokat

            maszk = np.zeros((magassag, szelesseg), dtype=np.uint8)
            cv2.fillPoly(maszk, [burok], FEHER)

            effekt = cv2.applyColorMap(kep, szinskala)
            kep[maszk == FEHER] = effekt[maszk == FEHER]

            cv2.polylines(kep, [burok], True, NEGYSZOG_SZIN, 3)

            # Felirat a keret bal felső sarka fölé
            x, y, _, _ = cv2.boundingRect(burok)
            cv2.putText(kep, f"PORTAL: {szuro_nev}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEGYSZOG_SZIN, 2)

        cv2.imshow("Portal", kep)
        if cv2.waitKey(1) & 0xFF == ord("q"):   # q gombbal kilépés
            break

kamera.release()
cv2.destroyAllWindows()