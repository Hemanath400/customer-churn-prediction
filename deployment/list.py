def remove_negatives(numbers):
    # Iterate through the list and remove negative values
    new_list=[]
    for x in numbers:
        if x>=0:
            new_list.append(x)
    return numbers


my_list = [3, -2, -5, 4, -1]
result = remove_negatives(my_list)
print(f"Result: {result}")
