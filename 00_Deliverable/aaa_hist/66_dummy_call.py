import os
import subprocess

if __name__ == "__main__":
    # Request input from the user
    input_int1 = input("Enter an integer: ")
    input_int2 = input("Enter an integer: ")
    
    # Call scriptA.py and pass the input as an argument
    #os.system('python 66_dummy_sub.py {} {}'.format(input_int1,input_int2))
    process = subprocess.Popen(['python', '66_dummy_sub.py', input_int1, input_int2])
    
    print("launched")
    process.wait()
    
    print('sub is done')
    