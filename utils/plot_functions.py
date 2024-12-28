""" 
The Plotting Module - Part of the `utils` library for Project Mjölnir

Developed by Selman Tabet @ https://selman.io/
----------------------------------------------
This module contains functions for evaluating models, aggregating results and plotting evaluation metrics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Switch to a non-interactive backend to allow for cmd line execution
plt.switch_backend('agg')


def plot_roc_curve(model_name, labels, predictions, directory, dataset_name):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for a given model and dataset, 
    saves the plot to a specified directory, and returns the optimal threshold.

    Arguments:
    ----------
        model_name (`str`): The name of the model being evaluated.
        labels (`array-like`): True binary labels.
        predictions (`array-like`): Probability predictions.
        directory (`str`): The directory where the ROC curve plot will be saved.
        dataset_name (`str`): The name of the dataset being evaluated.

    Returns:
    --------
        `float`: The optimal threshold value that maximizes the difference between true positive rate (TPR) and false positive rate (FPR).
    """

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
    plt.title(f'Receiver Operating Characteristic for {
              model_name} on {dataset_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, f'roc_curve_{
                model_name}_{dataset_name}.png'))
    plt.close()

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def plot_confusion_matrix(cm, directory, model_name, dataset_name, optimal=False):
    """
    Plots and saves a confusion matrix as a heatmap.

    Arguments:
    ----------
        cm (`array-like`): Confusion matrix to be plotted.
        directory (`str`): Directory where the plot will be saved.
        model_name (`str`): Name of the model.
        dataset_name (`str`): Name of the dataset.
        optimal (`bool`, optional): If `True`, indicates that the confusion matrix is optimal. Default is `False`.

    Returns:
    --------
        `None`
    """

    class_names = ['fire', 'nofire']
    # Save the confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    if optimal:
        plt.title(f'Optimal Confusion Matrix for {
                  model_name} on {dataset_name}')
    else:
        plt.title(f'Confusion Matrix for {model_name} on {dataset_name}')
    fname = f'confusion_matrix_{model_name}_{
        dataset_name}_optimal.png' if optimal else f'confusion_matrix_{model_name}_{dataset_name}.png'
    plt.savefig(os.path.join(directory, fname))
    plt.close()


def plot_pr_curve(model_name, labels, predictions, directory, dataset_name):
    """
    Plots the Precision-Recall (PR) curve for a given model and dataset, and saves the plot to a specified directory.
    Also calculates and returns the optimal threshold based on the maximum F1 score.

    Arguments:
    ----------
        model_name (`str`): The name of the model being evaluated.
        labels (`array-like`): True binary labels in range `{0, 1}`.
        predictions (`array-like`): Estimated probabilities or decision function.
        directory (`str`): The directory where the PR curve plot will be saved.
        dataset_name (`str`): The name of the dataset being evaluated.

    Returns:
    --------
        `float`: The optimal threshold based on the maximum F1 score.
    """

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
    plt.title(f'Precision-Recall Curve for {model_name} on {dataset_name}')
    plt.axhline(y=optimal_threshold, color='r', linestyle='--',
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, f'pr_curve_{
                model_name}_{dataset_name}.png'))
    plt.close()

    return optimal_threshold

# Function to plot and save training history


def plot_history(run_dir, history, model_name, dataset_name):
    """
    Plots and saves the training history of a machine learning model.

    Arguments:
    ----------
        run_dir (`str`): The directory where the plots and CSV file will be saved.
        history (`History`): The history object returned by the `fit` method of a Keras model.
        model_name (`str`): The name of the model.
        dataset_name (`str`): The name of the dataset.

    Returns:
    ------
        `None` - Saves the plots and CSV file to the specified directory.
    """

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
    plt.title(f'Training and Validation Accuracy for {
              model_name} on {dataset_name}')
    plt.legend()

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'Training and Validation Loss for {
              model_name} on {dataset_name}')
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
    plt.title(f'Training and Validation Recall for {
              model_name} on {dataset_name}')
    plt.legend()

    # Plot F1 score
    plt.subplot(3, 1, 2)
    plt.plot(epochs, f1, label='Training F1 Score')
    plt.plot(epochs, val_f1, label='Validation F1 Score')
    plt.title(f'Training and Validation F1 Score for {
              model_name} on {dataset_name}')
    plt.legend()

    # Plot precision
    plt.subplot(3, 1, 3)
    plt.plot(epochs, precision, label='Training Precision')
    plt.plot(epochs, val_precision, label='Validation Precision')
    plt.title(f'Training and Validation Precision for {
              model_name} on {dataset_name}')
    plt.legend()

    plt.savefig(os.path.join(run_dir, f"recall_f1_precision_{
                model_name}_{dataset_name}.png"))
    plt.close()


def plot_test_images(test_dataset, directory, dataset_name, model, threshold=0.5, optimal=False):
    """
    Plots a set of test images with their actual and predicted labels.

    Arguments:
    ----------
        test_dataset (`tf.data.Dataset`): The dataset containing test images and labels.
        directory (`str`): The directory where the plot image will be saved.
        dataset_name (`str`): The name of the dataset, used in the saved plot image filename.
        model (`tf.keras.Model`): The trained model used for making predictions.
        threshold (`float`, optional): The threshold for classifying predictions. Defaults to `0.5`.
        optimal (`bool`, optional): If `True`, appends '_optimal' to the saved plot image filename. Defaults to `False`.

    Returns:
    --------
        `None`
    """

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
