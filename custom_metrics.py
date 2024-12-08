import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable

# F1 score metric function modified from a GPT-4o example.


@register_keras_serializable()
def f1_score(y_true, y_pred):
    y_pred = K.flatten(y_pred)  # Flatten y_pred to match the shape of y_true

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_val = true_positives / (possible_positives + K.epsilon())
        return recall_val

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_val = true_positives / (predicted_positives + K.epsilon())
        return precision_val

    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))
