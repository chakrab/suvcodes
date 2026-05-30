import uvicorn
from uvicorn.config import LOGGING_CONFIG
from tinydb import TinyDB, Query
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse

class FaviconService:
    def __init__(self, name: str):
        self.name = name
        self.router = APIRouter()
        self.router.add_api_route("/favicon.ico", self.getfavicon, methods=["GET"], include_in_schema=False)

    def getfavicon(self):
        return FileResponse("./favicon.ico", media_type="image/x-icon")

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
        """
        Get the list of candidates for the given subject.
        """
        Resume = Query()
        table = self.tinydb.table('candidates')
        candidates = [record.get('candidate') for record in table.search(Resume.subject == subject)]
        return {"candidates": candidates}

    def get_resumes(self, subject: str):
        """
        Get the list of resumes for the given subject.
        """
        Resume = Query()
        table = self.tinydb.table('candidates')
        resumes = table.search(Resume.subject == subject)
        return {"resumes": resumes}

    def get_resume(self, candidate: str):
        """
        Get the resume for the given candidate name.
        """
        Resume = Query()
        table = self.tinydb.table('candidates')
        resume = table.search(Resume.candidate == candidate)
        return {"resume": resume}
    
    def get_subjects(self):
        """
        Get the list of available subjects.
        """
        table = self.tinydb.table('candidates')
        subjects = sorted(set(record.get('subject') for record in table.all()))
        return {"subjects": list(subjects)}

if __name__ == "__main__":
    app = FastAPI()
    favicon_viewer = FaviconService("Favicon Viewer")
    app.include_router(favicon_viewer.router)

    resume_service = ResumeService("Resume Service")
    app.include_router(resume_service.router, prefix="/api")
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    uvicorn.run(app, host="localhost", port=8000)