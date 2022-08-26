import sys

if len(sys.argv) != 2:
    print("Provide a number")
else:
    kbit = int(sys.argv[1])
    print(kbit >> 23)