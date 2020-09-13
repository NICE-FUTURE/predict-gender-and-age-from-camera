
def evaluate():
    log = open("./testset_result.log", "a", encoding="utf-8")
    with open("./testset_result.csv", "r", encoding="utf-8") as f:
        f.readline()
        correct = 0
        total = 0
        loss = 0
        for line in f:
            key, p_gender, g_gender, p_age, g_age = line.strip().split(",")
            if p_gender == g_gender:
                correct += 1
            loss += abs(float(p_age)-float(g_age))
            total += 1
        mae = loss/total
        accuracy = correct/total
        print("accuracy:{}, mae:{}".format(accuracy, mae))
        print("accuracy:{}, mae:{}".format(accuracy, mae), file=log)


if __name__ == "__main__":
    evaluate()
