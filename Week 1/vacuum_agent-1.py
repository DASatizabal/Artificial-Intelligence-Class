"""Unit 1: Modeling a Simple Vacuum-Cleaner Agent - Student Starter"""
from pathlib import Path
import random
import pandas as pd

def vacuum_agent(location, status):
    if status == "Dirty":
        return "SUCK"
    return "RIGHT" if location == "A" else "LEFT"

def run_two_room_simulation(environment="Dirty", seed=42, max_steps=8):
    random.seed(seed)
    if environment == "Clean":
        rooms={"A":"Clean","B":"Clean"}
    elif environment == "Dirty":
        rooms={"A":"Dirty","B":"Dirty"}
    else:
        rooms={"A":random.choice(["Clean","Dirty"]),"B":random.choice(["Clean","Dirty"])}
    location="A"; score=0
    for _ in range(max_steps):
        action=vacuum_agent(location, rooms[location])
        if action=="SUCK":
            if rooms[location]=="Dirty":
                rooms[location]="Clean"; score+=10
        else:
            location="B" if location=="A" else "A"; score-=1
        # TODO: Add stochastic recontamination.
        if rooms["A"]=="Clean" and rooms["B"]=="Clean": break
    return rooms,score

if __name__=="__main__":
    for scenario in ["Clean","Dirty","Stochastic"]:
        print(scenario, run_two_room_simulation(scenario))
    # TODO: Load the large CSV and compare environment performance.
