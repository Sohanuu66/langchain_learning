from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Annotated

class Student(BaseModel):
    name : str = 'sohannn'  # set default
    age : Optional[int] = None
    email : EmailStr
    cgpa : Annotated[float, Field(gt=0, le=10, default=5), "A decimal value representation"]

rahul = {'email' : 'abc123@gmail.com', 'cgpa' : 10}
# age = 32 / '32' valid coz python can implicitly do typeconversion
# age = 'sdfsdf'    pydantic validation error

student = Student(** rahul)

print(student.name)
print(type(student))
# <class '__main__.Student'> --> Pydantic object so access using .name 
student_dict = student.model_dump()
print(student_dict)

student_json = student.model_dump_json()
print(student_json)

