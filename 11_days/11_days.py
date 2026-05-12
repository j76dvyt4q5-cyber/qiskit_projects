import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import os
from models import db
from flask_login import LoginManager
from models import db, User
from routes.auth import auth_bp
from routes.main import main_bp

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'change-me-before-production'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
 
    db.init_app(app)
 
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
 
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
 
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
 
    with app.app_context():
        db.create_all()
 
    return app
 
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)



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

tasks = {
    '15 push ups',
    '15 sit ups',
    '30 sec plank',
    '20 min run',
    '15 squats',
    '15 min meditation',
    '30 min walk',
    '15 lunges',
    '5 min box breathing',
    '15 bicycles',
    '50 high-knees'
}

def add_new_task(day: dict):
    day['task'] = np.random.choice(list(tasks))

people = []
user_name = input("Enter your name: ")
user_age = int(input("Enter your age: "))
user_gender = input("Enter your gender: ")
if user_gender.lower().startswith('m'):
    user_gender = 'male'
elif user_gender.lower().startswith('f'):
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

user_profile = person(user_name, user_age, user_gender, user_height, user_weight, user_exercise_hours, user_exercise_class)

print(user_profile)

print("Your coach will now recommend your first workout:")

day_1 = {}
add_new_task(day_1)
 
print(f"    Your first workout is: {day_1['task']}, check in when you're done!")