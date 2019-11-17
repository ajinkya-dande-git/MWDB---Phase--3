from Main.Tasks.task3 import startTask3

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
            startTask3()
            pass
        elif int(userInput) == 2:
            pass
        elif int(userInput) == 3:
            pass
        elif int(userInput) == 4:
            pass
        elif int(userInput) == 5:
            pass
        elif int(userInput) == 6:
            pass
        else:
            exit()
