from .plot_functions import *
from sklearn.metrics import confusion_matrix


def full_eval(model_ds_dir, history, model, dataset_name, test_dataset, true_labels, steps):
    print(f"Evaluating {model.name} on {dataset_name}...")
    # Plot metrics and save the history
    plot_history(model_ds_dir, history, model.name, dataset_name)
    # Plot the test images with predictions
    plot_test_images(test_dataset, model_ds_dir, dataset_name, model)

    predicted_probs = model.predict(test_dataset, steps=steps)
    predicted_labels = (predicted_probs >= 0.5).astype(int).flatten()

    # Generate the confusion matrix using the default threshold of 0.5
    cm = confusion_matrix(true_labels, predicted_labels)
    plot_confusion_matrix(cm, model_ds_dir, model.name, dataset_name)

    # Generate the PR curve
    plot_pr_curve(model.name, true_labels, predicted_probs,
                  model_ds_dir, dataset_name)

    # Generate the ROC curve
    optimal_threshold = generate_roc_curve(
        model.name, true_labels, predicted_probs, model_ds_dir, dataset_name)

    # Plot test images with optimal threshold
    plot_test_images(test_dataset, model_ds_dir, dataset_name,
                     model, optimal_threshold, optimal=True)

    # Generate predictions with optimal threshold
    predicted_labels_optimal = (
        predicted_probs >= optimal_threshold).astype(int).flatten()

    # Generate the confusion matrix using the optimal threshold
    cm_optimal = confusion_matrix(true_labels, predicted_labels_optimal)
    plot_confusion_matrix(cm_optimal, model_ds_dir,
                          model.name, dataset_name, optimal=True)

    return optimal_threshold
