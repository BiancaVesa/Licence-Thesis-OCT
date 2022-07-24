
def predict_on_test_set(model, test_dir, img_height, img_width):
    """
    This function loads the test set and
    uses the given pretrained model to predict the labels.
    :param test_dir: the path of the directory of the test set
    :return: the true labels of the test set and the corresponding predictions.
    """

    images = glob.glob(test_dir)
    random.shuffle(images)
    CNV = "CNV"
    DME = "DME"
    DRUSEN = "DRUSEN"
    NORMAL = "NORMAL"

    predicted_categories = np.array([])
    true_categories = np.array([])
    for i in images:
        img = tf.keras.utils.load_img(i, target_size=(img_height, img_width, 3))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        prediction = model.predict(img_array)
        score = tf.nn.softmax(prediction[0])

        predicted_categories = np.concatenate([predicted_categories, [np.argmax(score)]])

        if CNV in i:
            true_categories = np.concatenate([true_categories, [0]])
        if DME in i:
            true_categories = np.concatenate([true_categories, [1]])
        if DRUSEN in i:
            true_categories = np.concatenate([true_categories, [2]])
        if NORMAL in i:
            true_categories = np.concatenate([true_categories, [3]])

    return true_categories, predicted_categories, images


def evaluate_model(true_categories, predicted_categories, class_names):
    """
    This function prints the classification report after prediction on test set
    and shows the confusion matrix.
    """

    print("Classification report:")
    print(classification_report(true_categories, predicted_categories, target_names=class_names))
    cm = confusion_matrix(true_categories, predicted_categories)
    print("Confusion matrix:")
    print(cm)