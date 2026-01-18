from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float

Base = declarative_base()

class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    cuisine = Column(String(100), nullable=False)
    rating = Column(Float, nullable=True)
    city = Column(String(100), nullable=False)
