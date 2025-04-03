# print(__name__) # quando lo lancio, mi dà main
def sum_things(a, b):
    return a + b

'''if __name__ == "__main__":
    print("ciao!")
else:
    print("ciao ciao!")'''
# lanciano direttamente questo viene ciao. Se lo importo e lo eseguo mi dà ciao ciao
# __name__ semplicemente è un controllo per vedere se lo lancio da quello principale o no

if __name__ == "__main__":
    print(sum_things(2, 7))