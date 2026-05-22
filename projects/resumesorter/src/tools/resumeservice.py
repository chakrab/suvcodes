import uvicorn
from uvicorn.config import LOGGING_CONFIG
from tinydb import TinyDB, Query
from fastapi import FastAPI, APIRouter

class ResumeService:
    """
    A simple resume service implemented using FastAPI and TinyDB.
    It provides endpoints to retrieve resumes based on subject and candidate name.
    """
    def __init__(self, name: str):
        self.name = name
        self.router = APIRouter()
        self.router.add_api_route("/", self.health, methods=["GET"])
        self.router.add_api_route("/get-candidates/{subject}", self.get_candidates, methods=["GET"])
        self.router.add_api_route("/get-resumes/{subject}", self.get_resumes, methods=["GET"])
        self.router.add_api_route("/get-resume/{candidate}", self.get_resume, methods=["GET"])
        self.router.add_api_route("/get-subjects", self.get_subjects, methods=["GET"])

        self.tinydb = TinyDB('../../data/db.json')

    def health(self):
        return {"status": "ok"}

    def get_candidates(self, subject: str):
        Resume = Query()
        table = self.tinydb.table('candidates')
        candidates = [record.get('candidate') for record in table.search(Resume.subject == subject)]
        return {"candidates": candidates}

    def get_resumes(self, subject: str):
        Resume = Query()
        table = self.tinydb.table('candidates')
        resumes = table.search(Resume.subject == subject)
        return {"resumes": resumes}

    def get_resume(self, candidate: str):
        Resume = Query()
        table = self.tinydb.table('candidates')
        resume = table.search(Resume.candidate == candidate)
        return {"resume": resume}
    
    def get_subjects(self):
        table = self.tinydb.table('candidates')
        subjects = sorted(set(record.get('subject') for record in table.all()))
        return {"subjects": list(subjects)}

if __name__ == "__main__":
    app = FastAPI()
    resume_service = ResumeService("Resume Service")
    app.include_router(resume_service.router, prefix="/api")
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    uvicorn.run(app, host="localhost", port=8000)