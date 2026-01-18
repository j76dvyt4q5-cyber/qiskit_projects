from states import ket0, ket1
from gates import I, X, Y, Z, H

def show(label, state):
    print(f"\n{label}:")
    print(state.draw("text"))
    print("Probabilities:", state.probabilities_dict())

#Basic Tests 
    #(gates to |0> state)
    show("I |0>", I @ ket0)
    show("X |0>", X @ ket0)
    show("Y |0>", Y @ ket0)
    show("Z |0>", Z @ ket0)
    show("H |0>", H @ ket0)

    #(gates to |1> state)
    show("I |1>", I @ ket1)
    show("X |1>", X @ ket1)
    show("Y |1>", Y @ ket1)
    show("Z |1>", Z @ ket1)
    show("H |1>", H @ ket1)

#Composition tests
    show("H H |0>", H @ H @ ket0)
    show("Z H |0>", Z @ H @ ket0)

psi = H @ ket0
print(psi.draw("text"))

print("H shape:", H.input_dims(), H.output_dims())
print("ket0 shape:", ket0.dim)



