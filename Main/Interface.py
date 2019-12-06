from Main.Tasks.task1 import startTask1
from Main.Tasks.task2 import startTask2
from Main.Tasks.task3 import startTask3
from Main.Tasks.task4 import startTask4
from Main.Tasks.task5 import startTask5
from Main.Tasks.task6 import startTask6

if __name__ == '__main__':
    runAgain = True
    while runAgain:
        print("Select one of the below")
        print("1. Task 1")
        print("2. Task 2")
        print("3. Task 3")
        print("4. Task 4")
        print("5. Task 5")
        print("6. Task 6")
        print("Any other number to exit")
        userInput = input()
        if int(userInput) == 1:
            startTask1()
        elif int(userInput) == 2:
            startTask2()
        elif int(userInput) == 3:
            startTask3()
        elif int(userInput) == 4:
            startTask4()
        elif int(userInput) == 5:
            startTask5()
        elif int(userInput) == 6:
            startTask6()
        else:
            exit()
