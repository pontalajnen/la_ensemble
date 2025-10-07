from argparse import ArgumentParser
import os 


def print_info(script, text):     
    script = script.split("/")[-1]     
    padding = (max([len(file) for file in os.listdir(".")]) - len(script)) * " "
    print(f"[{script}] {padding} ----- {text} -----")
