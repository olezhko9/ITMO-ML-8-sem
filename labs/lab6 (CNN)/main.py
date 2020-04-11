from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
NUM_CLASSES = 10

print(train_images.shape)
print(test_images.shape)

train_images = train_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images / 255.0
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

best_model = None
best_acc = 0.0

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1, verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nError Rate на проверочных данных :', 1 - test_acc)

if test_acc > best_acc: best_model = model


def confusion_matrix(actual, pred):
    cm = [[0 for j in range(NUM_CLASSES)] for i in range(NUM_CLASSES)]
    most_similar = [(-1, 0.0) for i in range(NUM_CLASSES * NUM_CLASSES)]

    for (i, labels_prob) in enumerate(pred):
        label = K.argmax(labels_prob)
        cm[actual[i]][label] += 1

        if labels_prob[label] > most_similar[actual[i] * NUM_CLASSES + label][1]:
            most_similar[actual[i] * NUM_CLASSES + label] = (i, labels_prob[label])

    return cm, most_similar


pred_labels = best_model.predict(test_images)

class_names = ['Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто', 'Сандали', 'Блузка', 'Кроссовки', 'Сумка',
               'Ботильоны']
print(class_names)
cm, most_similar = confusion_matrix(test_labels, pred_labels)
print(pd.DataFrame(cm))

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
blank_image = [[255 for j in range(28)] for i in range(28)]
plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.8, hspace=0.2)
for i in range(NUM_CLASSES * NUM_CLASSES):
    plt.subplot(10, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if most_similar[i][0] == -1:
        plt.imshow(blank_image, cmap=plt.cm.binary)
        continue
    plt.imshow(test_images[most_similar[i][0]], cmap=plt.cm.binary)
    plt.xlabel(class_names[i % NUM_CLASSES])
plt.show()
