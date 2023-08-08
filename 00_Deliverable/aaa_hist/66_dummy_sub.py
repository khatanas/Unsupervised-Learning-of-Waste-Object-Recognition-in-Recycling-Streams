import sys
import time 

def double_input(some_input):
    return some_input * 2

if __name__ == "__main__":
    time.sleep(2)
    # Calculate the double of the input
    result1 = double_input(int(sys.argv[1]))
    result2 = double_input(int(sys.argv[2]))
    
    
    # Print the result
    print(f"The double of {sys.argv[1]} is {result1}.")
    print(f"The double of {sys.argv[2]} is {result2}.")
