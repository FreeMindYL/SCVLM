import re
import os
# import torch
import numpy as np
import json


# Position the call.value to generate the graph
def extract_pattern(code, func_list):
    # allFunctionList = split_function(code)  # Store all functions
    allFunctionList = func_list
    callValueList = []  # Store all functions that call call.value
    otherFunctionList = []  # Store functions other than the functions that contains call.value
    pattern_list = [0,0,0]

    # Store functions other than W functions (with .call.value)
    for i in range(len(allFunctionList)):
        # print(f"{i}-----")     # <<================================
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            # print(allFunctionList[i][j])
            if '.call.value' in text:
                callValueList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    ################  pattern 1: callValueInvocation  #######################
    if len(callValueList) != 0:
        pattern_list[0] = 1


    ################   pattern 2: balanceDeduction   #######################
    for i in range(len(callValueList)):
        CallValueFlag1 = 0

        for j in range(len(callValueList[i])):
            text = callValueList[i][j]
            if '.call.value' in text:
                CallValueFlag1 += 1
            elif CallValueFlag1 != 0:
                text = text.replace(" ", "")
                if "-" in text or "-=" in text or "=0" in text:
                    pattern_list[1] = 1
                    break


    ################   pattern 3: enoughBalance     #######################
    for i in range(len(callValueList)):
        CallValueFlag2 = 0
        key_param = []
        key_word_idx = 0

        for j in range(len(callValueList[i])):
            text = callValueList[i][j]
            if '.call.value' in text:
                key_word_idx = j
                CallValueFlag2 += 1
                # print(text)
                param = re.findall(r".call.value\((.+?)\)", text)
                key_param += param
 
        if CallValueFlag2 == 0:
            pass
        else:
            check_code = callValueList[i][:key_word_idx]  
            for text in check_code:
                for param in key_param:
                    # print(param)
                    # print(text)
                    if (param in text) and ('.call.value' not in text):
                        if '>' in text or '<' in text:
                            pattern_list[2] = 1
                            break

        
            # elif CallValueFlag2 != 0:
            #     # if(type(param)) == list:
            #     #     str(param)
            #     if param in text:
                    
            #         pattern_list.append(1)
            #         break
            #     elif j + 1 == len(callValueList[i]) and len(pattern_list) == 2:
            #         pattern_list.append(0)

    return pattern_list



def reen_gen_pattern(code, func_list):
    pattern_list = extract_pattern(code, func_list)
    pattern01 = pattern_list

    pattern = ['safe', 'reentrancy']
    if pattern01[0] == 1:
        pattern.append("keyword")         #Keyword
    else:
        pattern.append("None")

    if pattern01[1] == 1:
        pattern.append("deduction")   # Invocation   Assignment
    else:
        pattern.append("None")
    
    if pattern01[2] == 1:
        pattern.append("checkpoint")
    else:
        pattern.append("None")

    return pattern

