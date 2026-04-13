import json
import random

titles = ["Backend Developer", "Frontend Developer", "Data Analyst", "ML Engineer"]
companies = ["TCS", "Infosys", "Wipro", "Google", "Amazon"]
locations = ["Pune", "Bangalore", "Hyderabad", "Mumbai"]
skills_pool = ["Python", "Java", "SQL", "React", "Docker", "AWS"]

jobs = []

for i in range(50):
    job = {
        "id": i+1,
        "title": random.choice(titles),
        "company": random.choice(companies),
        "location": random.choice(locations),
        "experience": f"{random.randint(0,3)}-{random.randint(3,6)} years",
        "skills": random.sample(skills_pool, 3),
        "description": "Responsible for development and maintenance of applications."
    }
    jobs.append(job)

with open("jobs.json", "w") as f:
    json.dump(jobs, f, indent=4)
    
print(jobs[1])