from logic_gate import AND
from logic_gate import OR
from logic_gate import XOR
from logic_gate import NOT


And = AND()
Or = OR()
Not = NOT()
Xor = XOR()

print("not True is: ",Not(True))
print("not False is: ",Not(False))

print("-----------------------------")

print("True and True is: ",And(True,True))
print("True and False is: ",And(True,False))
print("False and True is: ",And(False,True))
print("False and False is: ",And(False,False))

print("-----------------------------")

print("True or True is: ",Or(True,True))
print("True or False is: ",Or(True,False))
print("False or True is: ",Or(False,True))
print("False or False is: ",Or(False,False))

print("-----------------------------")

print("True xor True is: ",Xor(True,True))
print("True xor False is: ",Xor(True,False))
print("False xor True is: ",Xor(False,True))
print("False xor False is: ",Xor(False,False))