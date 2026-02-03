from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

rahul : Person = {'name' : 'rahul', 'age' : 'slakdfj'}
# it doesnt return any error on age being a string or an integer as 
# typeddict only provides type hints and nothing else --> no error on run
# but type hints will be used by tools that helps in --> structured output

print(rahul)