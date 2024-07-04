from ctypes import CDLL, c_int, POINTER
import numpy as np

# definizione del dll
c_code = CDLL('./script.dll')

# assegnazione dei tipi alla funzione del dll
c_code.function.argtypes = [POINTER(c_int), c_int]
c_code.function.restype = c_int

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# trasformazione della lista nell'array da passare come argomento alla funzione di c
arg_array = np.array(lst).ctypes.data_as(POINTER(c_int))

# esecuzione del codice c
res = c_code.function(arg_array, len(lst))

print(res)