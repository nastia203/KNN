import csv
import math
import heapq


def load_csv(file):
    iris_set = []
    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            iris_set.append(row)
    return iris_set


def euclidean_distance(row1, row2):
    dist = 0
    for i in range(len(row1) - 1):
        dist += math.sqrt((float(row1[i]) - float(row2[i])) ** 2)
    return dist


def get_neighbors(train, t_row, k=1):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(t_row, train_row)
        heapq.heappush(distances, (dist, train_row))
    neighbors = [heapq.heappop(distances)[1] for _ in range(k)]
    return neighbors


def predict_class(train, t_row, num):
    neighbors = get_neighbors(train, t_row, num)
    value = [row[-1] for row in neighbors]
    prediction = max(set(value), key=value.count)
    return prediction


def knn(train_file, test_file, k):
    train = load_csv(train_file)
    test = load_csv(test_file)
    pred = []
    for row in test:
        output = predict_class(train, row, k)
        pred.append(output)
    actual = load_csv(test_file)
    acc = accuracy(actual, pred)
    return acc, pred


def accuracy(act, predict):
    correct = 0
    for i in range(len(act)):
        if act[i][-1] == predict[i]:
            correct += 1
    print(f"All classifications: {len(act)}")
    print(f"Correct classifications: {correct}")
    return correct / float(len(act)) * 100.0


def single(train_file, arr, k):
    train = load_csv(train_file)
    output = predict_class(train, arr, k)
    return output


if __name__ == "__main__":
    train_set = 'Train-set.csv'
    test_set = 'Test-set.csv'
    while (True):
        command = input("1)Specify vector for classification\n"
                        "2)Number of precise classifications\n"
                        "3)Exit\n"
                        "Enter command:\n ")
        match command:
            case "1":
                k = int(input('Enter the number of neighbors: '))
                arr_input = input('Enter the vector in the form [0.0,0.0,0.0,0.0]: ')
                arr = [float(i) for i in arr_input.split(",")]
                print(f"Predicted class: {single(train_set, arr, k)}")
            case "2":
                k = int(input("Enter the number of neighbors:"))
                acc, predictions = knn(train_set, test_set, k)
                print(f'The accuracy of this classification: {acc:.3f}%')
            case "3":
                break
            case _:
                print("Invalid command.")
