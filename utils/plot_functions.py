import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Switch to a non-interactive backend to allow for cmd line execution
plt.switch_backend('agg')


def plot_roc_curve(model_name, labels, predictions, directory, dataset_name):
    # Generate predictions on the validation set

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, f'roc_curve_{
                model_name}_{dataset_name}.png'))
    plt.close()

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def plot_confusion_matrix(cm, directory, model_name, dataset_name, optimal=False):
    class_names = ['fire', 'nofire']
    # Save the confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    fname = f'confusion_matrix_{model_name}_{
        dataset_name}_optimal.png' if optimal else f'confusion_matrix_{model_name}_{dataset_name}.png'
    plt.savefig(os.path.join(directory, fname))
    plt.close()


def plot_pr_curve(model_name, labels, predictions, directory, dataset_name):
    # Generate predictions on the validation set
    precision, recall, thresholds = precision_recall_curve(
        labels, predictions)

    # Optimizing based on max F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Plot PR curve
    plt.figure()
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, f'pr_curve_{
                model_name}_{dataset_name}.png'))
    plt.close()

    return optimal_threshold

# Function to plot and save training history


def plot_history(run_dir, history, model_name, dataset_name):
    # Ensure all arrays in history.history have the same length
    min_length = min(len(values) for values in history.history.values())
    history_dict = {key: values[:min_length]
                    for key, values in history.history.items()}
    history_df = pd.DataFrame(history_dict)
    history_df.to_csv(os.path.join(run_dir, f"history_{
                      model_name}_{dataset_name}.csv"), index=False)
    # Accuracy and Loss Metrics
    acc = history_df['accuracy']
    val_acc = history_df['val_accuracy']
    loss = history_df['loss']
    val_loss = history_df['val_loss']
    precision = history_df['precision']
    val_precision = history_df['val_precision']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 12))

    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig(os.path.join(run_dir, f"loss_accuracy_{
                model_name}_{dataset_name}.png"))
    plt.close()

    # Recall, F1 Score, and Precision Metrics
    recall = history.history['recall']
    val_recall = history.history['val_recall']
    f1 = history.history['f1_score']
    val_f1 = history.history['val_f1_score']
    precision = history.history['precision']
    val_precision = history.history['val_precision']
    epochs = range(len(recall))

    plt.figure(figsize=(12, 18))

    # Plot recall
    plt.subplot(3, 1, 1)
    plt.plot(epochs, recall, label='Training Recall')
    plt.plot(epochs, val_recall, label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.legend()

    # Plot F1 score
    plt.subplot(3, 1, 2)
    plt.plot(epochs, f1, label='Training F1 Score')
    plt.plot(epochs, val_f1, label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    # Plot precision
    plt.subplot(3, 1, 3)
    plt.plot(epochs, precision, label='Training Precision')
    plt.plot(epochs, val_precision, label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.legend()

    plt.savefig(os.path.join(run_dir, f"recall_f1_precision_{
                model_name}_{dataset_name}.png"))
    plt.close()


def plot_test_images(test_dataset, directory, dataset_name, model, threshold=0.5, optimal=False):
    while True:
        try:
            test_images, test_labels = next(iter(test_dataset))
            predictions = model.predict(test_images)

            fire_indices = np.where(test_labels == 1)[0]
            nofire_indices = np.where(test_labels == 0)[0]

            random_fire_indices = np.random.choice(
                fire_indices, 5, replace=False)
            random_nofire_indices = np.random.choice(
                nofire_indices, 5, replace=False)

            random_indices = np.concatenate(
                (random_fire_indices, random_nofire_indices))
            np.random.shuffle(random_indices)

            # Plot the images with predictions
            plt.figure(figsize=(20, 10))
            for i, idx in enumerate(random_indices):
                plt.subplot(2, 5, i+1)
                plt.imshow(test_images[idx])
                plt.title(f"Actual: {'No Fire' if test_labels[idx] == 1 else 'Fire'}\nPredicted: {
                          'No Fire' if predictions[idx] >= threshold else 'Fire'}")
                plt.axis('off')
            plt.savefig(os.path.join(directory, f"test_images_{model.name}_{
                        dataset_name}_optimal.png" if optimal else f"test_images_{model.name}_{dataset_name}.png"))
            plt.close()
            break

        except ValueError:
            pass
