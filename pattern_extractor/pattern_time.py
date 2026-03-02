import os
import torch
import numpy as np


# Position the call.value to generate the graph
def extract_pattern(code, func_list):
    # allFunctionList = split_function(code)  # Store all functions
    allFunctionList = func_list
    timeStampList = []  # Store all W functions that call call.value
    otherFunctionList = []  # Store functions other than W functions
    pattern_list = [0,0,0]

    # Store other functions without W functions (with block.timestamp)
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if 'block.timestamp' in text:
                timeStampList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    ################   pattern 1: timestampInvocation  #######################
    if len(timeStampList) != 0:
        pattern_list[0] = 1


    ################   pattern 2: timestampAssign      #######################
    for i in range(len(timeStampList)):
        TimestampFlag1 = 0
        VarTimestamp = []

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]
            if 'block.timestamp' in text:
                TimestampFlag1 += 1
                if '=' in text:
                    value = text.split("=")[0]
                    value = value.strip()
                    key_value = value.split(" ")[-1]
                    VarTimestamp.append(key_value)

            elif TimestampFlag1 != 0 and len(VarTimestamp) >0:
                for key_var in VarTimestamp: 
                    if key_var in text:
                        pattern_list[1] = 1
                        break

    ################  pattern 3: timestampContamination  #######################
    for i in range(len(timeStampList)):
        TimestampFlag2 = 0
        VarTimestamp2 = []

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]
            # print(text)
            if 'block.timestamp' in text:
                if '=' in text:
                    value = text.split("=")[0]
                    value = value.strip()
                    key_value = value.split(" ")[-1]
                    VarTimestamp2.append(key_value)
                TimestampFlag2 += 1
                if 'return' in text:
                    pattern_list[2] = 1
                    break
            elif TimestampFlag2 != 0 and len(VarTimestamp2)>0:
                for key_var in VarTimestamp2: 
                    if key_var in text:
                        if 'return' in text:
                            pattern_list[2] = 1
                            break

    return pattern_list


def time_gen_pattern(code, func_list):
    pattern_list = extract_pattern(code, func_list)
    pattern01 = pattern_list

    pattern = ['safe', 'timestamp dependency']
    if pattern01[0] == 1:
        pattern.append("keyword")       #Keyword
    else:
        pattern.append("None")

    if pattern01[1] == 1:
        pattern.append("assignment")   # Invocation   Assignment
    else:
        pattern.append("None")
    
    if pattern01[2] == 1:
        pattern.append("return")
    else:
        pattern.append("None")
    

    return pattern

