import sys

def main():
    gold_file = open(sys.argv[1], 'r')
    student_file = open(sys.argv[2], 'r')


    # first line
    gold_line = gold_file.readline()
    student_line = student_file.readline()
    print(student_line.split("\t"))
    if len(student_line.split("\t")) != 3 and len(student_line.split("\t")) != 5:
        print("Wrong format at line 0")
        return

    # all lines
    for i in range(1,40):
        gold_line = gold_file.readline()
        student_line = student_file.readline()

        gold_output = gold_line.split("\t")
        student_output = student_line.split("\t")

        if len(gold_output) != len(student_output):
            print(f"Wrong format at line {i}")
            return
        
        if gold_output[0] != student_output[0]:
            print(f"Wrong format at line {i}")
            return

    print("A valid format!")

    gold_file.close()
    student_file.close()
            


if __name__ == '__main__':
    main()