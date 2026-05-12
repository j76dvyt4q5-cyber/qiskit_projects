from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
 
db = SQLAlchemy()
 
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name     = db.Column(db.String(80))