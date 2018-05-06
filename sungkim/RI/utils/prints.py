import os
import time

'''
Print Utils
'''

# Clear console
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def print_frozenlake_result(score):
    """Prints GOAL if score is positive else DEAD"""
    message = "GOAL" if score > 0 else "DEAD"
    print("=" * 50)
    print("{:^50}".format(message))
    print("=" * 50)
    time.sleep(3)