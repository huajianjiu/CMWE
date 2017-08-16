import sys

# USEFUL if you want to set model check point
with open(sys.argv[1], "r") as f:
    best_loss = 10
    best_acc = 0
    for line in f.readlines():
        if "DATASET" in line:
            print("best loss: ", best_loss, " best acc: ", best_acc, end="\n\n")
            print(line)
        if "Only" in line or "HATT" in line or "fasttext" in line:
            print("best loss: ", best_loss, " best acc: ", best_acc, end="\n\n")
            print(line, end="")
            best_loss = 10
            best_acc = 0
        if "val_loss" in line:
            val_loss = float(line[line.find("val_loss") + 10:line.find("val_loss") + 15])
            if val_loss < best_loss:
                best_loss = val_loss
        if "val_acc" in line:
            val_acc = float(line[line.find("val_acc") + 10:line.find("val_acc") + 15])
            if val_acc > best_acc:
                best_acc = val_acc
    print("best loss: ", best_loss, " best acc: ", best_acc, end="\n\n")