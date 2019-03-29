from speechemotionrecognition.mlmodel import NN
from speechemotionrecognition.utilities import get_data


data_path = 'dataset'

class_labels = ["Neutral", "Angry", "Happy", "Sad"]


def ml_example():
    x_train, x_test, y_train, y_test = get_data(data_path=data_path)
    model = NN()
    print('Starting', model.name)
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    print(model.name, 'Done')


if __name__ == "__main__":
    ml_example()
