import numpy as np
import app.dodecaMarkerCorners as dodecaMarkerConers

#TODO: Update these
IMU_OFFSET = (0.0, -0.01, 0.004)  # position of IMU relative to the top of the stylus
STYLUS_LENGTH = 0.1686  # length from the tip to the top of the stylus

def rotateY(angle: float, point: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.dot(rotation_matrix, point)


# markerLength = 0.015
def getMarkerCorners(markerLength: float):
    return np.array(
        [
            [-markerLength / 2, markerLength / 2, 0],
            [markerLength / 2, markerLength / 2, 0],
            [markerLength / 2, -markerLength / 2, 0],
            [-markerLength / 2, -markerLength / 2, 0],
        ],
        dtype=np.float32,
    )


def getCornersPS(
    origin: np.ndarray, angleY: float, markerLength: float = 0.013*0.97 #TODO: update markerLength here
) -> np.ndarray:
    cornersWS = getMarkerCorners(markerLength) + origin
    rotated_corners = np.apply_along_axis(lambda x: rotateY(angleY, x), 1, cornersWS)
    return rotated_corners - IMU_OFFSET


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180

idealMarkerPositions = {}

for i in range(12):
    idealMarkerPositions[i+1] = dodecaMarkerConers.corners_in_cart_sp[i]

# Assign the matrix at index i to the dictionary with key i+1
# { #TODO: update these with dodeca ball data
#     99: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(135)),
#     98: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(225)),
#     97: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(315)),
#     96: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(45)),
#     95: getCornersPS(np.array([0, -0.0395, 0.01], dtype=np.float32), deg2rad(90)),
#     94: getCornersPS(np.array([0, -0.0395, 0.01], dtype=np.float32), deg2rad(180)),
#     93: getCornersPS(np.array([0, -0.0395, 0.01], dtype=np.float32), deg2rad(270)),
#     92: getCornersPS(np.array([0, -0.0395, 0.01], dtype=np.float32), deg2rad(0)),
# }