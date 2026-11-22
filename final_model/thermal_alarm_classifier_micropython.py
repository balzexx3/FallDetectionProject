
import math

FEATURE_NAMES = [
    "frame_difference_mean",
    "frame_difference_max",
    "center_displacement_mean",
    "center_displacement_max",
    "hot_pixels_mean",
    "hot_pixels_max",
    "hot_pixels_variation_mean"
]

# Missing-value imputation medians
MEDIANS = [
    0.4607449,
    0.59866756,
    1.4512307457811076,
    3.5776593239065995,
    55.1875,
    67.0,
    4.695652173913044
]

# StandardScaler parameters
MEANS = [
    0.46350076354378816,
    0.6053173572165649,
    1.4866095662043899,
    3.7066060578901436,
    54.85679186006031,
    66.8499660556687,
    4.812578255950392
]

SCALES = [
    0.015306930169912079,
    0.06430562402689877,
    0.3660070322488994,
    1.094755513601691,
    6.180780963378275,
    7.542315228606751,
    1.1084100262419014
]

# Logistic Regression parameters
COEFFICIENTS = [
    -0.29840498604769683,
    0.31263882182733554,
    -0.2701603003847054,
    0.6492704558673862,
    0.370939234460283,
    0.415141352513747,
    -0.010964586101187515
]

INTERCEPT = -0.21386376864279788


def predict_probability(features):
    """
    Input:
        features: list containing exactly 7 values
                  in FEATURE_NAMES order.

    Output:
        Probability that the candidate alarm
        corresponds to a real fall.
    """

    if len(features) != 7:
        raise ValueError("Expected exactly 7 features")

    z = INTERCEPT

    for i in range(7):

        value = features[i]

        # Handle missing values
        if value is None:
            value = MEDIANS[i]
        else:
            value = float(value)

            if math.isnan(value):
                value = MEDIANS[i]

        # Standardization
        normalized = (value - MEANS[i]) / SCALES[i]

        # Logistic Regression
        z += COEFFICIENTS[i] * normalized

    # Numerically stable sigmoid
    if z >= 0:
        probability = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        probability = exp_z / (1.0 + exp_z)

    return probability
