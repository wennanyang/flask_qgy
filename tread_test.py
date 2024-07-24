from concurrent.futures import ThreadPoolExecutor

excut = ThreadPoolExecutor(3) 
string = ', '.join(str(i) for i in range(4))
print(string)
print("0, 1, 2, 3, 4")