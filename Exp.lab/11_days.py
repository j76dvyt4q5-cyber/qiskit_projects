class person:
    def __init__(self, name, age, gender, height, weight, exercise_hours, exercise_class):
        self.name = name
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.exercise_hours = exercise_hours
        self.exercise_class = exercise_class
        return None
    
    def __str__(self):
        return f"Name: {self.name} | Age: {self.age} | Gender: {self.gender} | Height: {self.height} inches | Weight: {self.weight} pounds | Exercise hours: {self.exercise_hours} | Exercise class: {self.exercise_class}"

people = []
user_name = input("Enter your name: ")
user_age = int(input("Enter your age: "))
user_gender = input("Enter your gender: ")
if "M" in user_gender:
    user_gender = 'male'
elif "F" in user_gender or "f" in user_gender:
    user_gender = 'female'

user_height = float(input("Enter your height in inches: "))
user_weight = float(input("Enter your weight in pounds: "))
user_exercise_hours = int(input("On average, how many hours of exercise do you get per week? "))
if user_exercise_hours >= 12:
    user_exercise_class = 'high'
elif user_exercise_hours >= 6:
    user_exercise_class = 'medium'
elif 6 > user_exercise_hours > 0:
    user_exercise_class = 'low'
else: 
    user_exercise_class = 'none'

user_profile = person(user_name, user_age, user_gender, user_weight, user_height, user_exercise_hours, user_exercise_class)

print(user_profile)

print("Your coach will now recommend your first workout")

day_1 = {
    'task 1' : '20 push-ups',
    'task 2' : '20 sit-ups',
    'task 3' : '20 squats'
}

day_2 = {
    'task 4' : '20 lunges'
}

Tasks = {
    'day 1' : ['task 1', 'task 2', 'task 3'],
    'day 2' : ['task 1', 'task 2', 'task 3', 'task 4'],
}

print(f"Your first workout is: {day_1['task 1']}, check in when you're done!")