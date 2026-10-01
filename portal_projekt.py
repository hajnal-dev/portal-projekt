import cv2
import mediapipe as mp
import numpy as np

mp_kezek = mp.solutions.hands          # a kézfelismerő modul
mp_rajz = mp.solutions.drawing_utils   # segédfüggvény a pontok kirajzolásához

HUVELYK_HEGY = 4    # a 21 landmark közül a hüvelykujj hegye
MUTATO_HEGY = 8     # a mutatóujj hegye
CSIPES_KUSZOB = 0.05  # ennél kisebb távolság = csípés (normalizált egységben)
NEGYSZOG_SZIN = (0, 255, 0)  # BGR: zöld


def pixelbe(pont, szelesseg, magassag):
    """Normalizált (0-1) landmarkot pixelkoordinátává alakít, (x, y) tuple-ként."""
    return (int(pont.x * szelesseg), int(pont.y * magassag))


kamera = cv2.VideoCapture(0)  # 0 = alapértelmezett webkamera

with mp_kezek.Hands(max_num_hands=2, min_detection_confidence=0.7) as kezek:
    while True:
        sikerult, kep = kamera.read()   # egy képkocka beolvasása
        if not sikerult:
            break

        kep = cv2.flip(kep, 1)  # tükrözés
        rgb = cv2.cvtColor(kep, cv2.COLOR_BGR2RGB)
        eredmeny = kezek.process(rgb)

        # A kép mérete minden kéznél ugyanaz, ezért elég egyszer lekérdezni
        magassag, szelesseg, _ = kep.shape

        sarkok = []  # ide gyűjtjük ennek a képkockának a sarokpontjait

        if eredmeny.multi_hand_landmarks:          # találtunk-e kezet?
            for kez in eredmeny.multi_hand_landmarks:
                mp_rajz.draw_landmarks(kep, kez, mp_kezek.HAND_CONNECTIONS)

                huvelyk = kez.landmark[HUVELYK_HEGY]
                mutato = kez.landmark[MUTATO_HEGY]

                tavolsag = np.linalg.norm(np.array([huvelyk.x, huvelyk.y]) - np.array([mutato.x, mutato.y]))
                if tavolsag <= CSIPES_KUSZOB:
                    cv2.putText(kep, "CSIPES!", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Kezenként két sarok: a hüvelykujj és a mutatóujj hegye
                sarkok.append(pixelbe(huvelyk, szelesseg, magassag))
                sarkok.append(pixelbe(mutato, szelesseg, magassag))

                # Csak akkor rajzolunk, ha mindkét kéz megvan (2 kéz × 2 pont)
        
        if len(sarkok) == 4:
            pontok = np.array(sarkok, dtype=np.int32)
            burok = cv2.convexHull(pontok)   # sorba rendezi a sarkokat

            # 1. Fekete maszk: ugyanakkora, mint a kép, de csak 1 színcsatornával
            maszk = np.zeros((magassag, szelesseg), dtype=np.uint8)

            # 2. A négyszög belsejét kifestjük fehérre → ez lesz a "bent"
            cv2.fillPoly(maszk, [burok], 255)

            # 3. Az egész képből elkészítjük a színeffektes változatot
            effekt = cv2.applyColorMap(kep, cv2.COLORMAP_PLASMA)

            # 4. Csak ott cseréljük le a pixeleket, ahol a maszk fehér
            kep[maszk == 255] = effekt[maszk == 255]
            cv2.imshow("Maszk", maszk)

            # 5. A keretet a végén rajzoljuk, hogy az effekt ne fesse felül
            cv2.polylines(kep, [burok], True, NEGYSZOG_SZIN, 3)
        cv2.imshow("Portal", kep)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):   # q gombbal kilépés
            break

kamera.release()
cv2.destroyAllWindows()