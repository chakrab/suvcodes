import uvicorn
import logging
from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse

from pathlib import Path

class ResumeFileService:
    def __init__(self, name: str, file_path: Path):
        self.file_path = file_path
        self.name = name
        self.router = APIRouter()
        self.router.add_api_route("/", self.health, methods=["GET"])
        self.router.add_api_route("/api/get-candidates/{subject}", self.get_candidates, methods=["GET"])
        self.router.add_api_route("/api/get-resume/{subject}/{candidate}", self.get_resume, methods=["GET"])
        self.router.add_api_route("/api/get-subjects", self.get_subjects, methods=["GET"])
        self.router.add_api_route("/favicon.ico", self.getfavicon, methods=["GET"], include_in_schema=False)

    def health(self):
        logging.info("Health check endpoint called")
        return {"status": "ok"}
    
    def get_candidates(self, subject: str):
        logging.info(f"Get candidates endpoint called with subject: {subject}")
        candidate_ids = self._get_candidate_ids(self.file_path, subject)
        return {"candidates": candidate_ids}
    
    def get_resume(self, subject: str, candidate: str):
        logging.info(f"Get resume endpoint called with subject: {subject}, candidate: {candidate}")
        resume_content = self._read_resume(self.file_path, candidate, subject)
        return {"resume": resume_content}

    def get_subjects(self):
        logging.info("Get subjects endpoint called")
        categories = self._get_categories(self.file_path)
        return {"subjects": categories}   
    
    def getfavicon(self):
        logging.info("Get favicon endpoint called")
        return FileResponse("./favicon.ico", media_type="image/x-icon")

    @staticmethod
    def _get_categories(file_path: Path):
        categories = set()
        for name in file_path.iterdir():
            if name.is_dir():
                categories.add(name.name)
        return list(categories) 
    
    @staticmethod
    def _get_candidate_ids(file_path: Path, category: str):
        category_path = file_path / category.upper()
        candidate_ids = set()
        for name in category_path.iterdir():
            if name.is_file() and name.suffix == ".md":
                candidate_ids.add(name.stem)
        return list(candidate_ids)

    @staticmethod
    def _read_resume(file_path: Path, candidate_id: str, category: str):
        file_path = file_path / category.upper() / f"{candidate_id}.md"
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return "Resume not found"
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"

if __name__ == "__main__":
    app = FastAPI()
    file_path = Path("../../data/md/")
    service = ResumeFileService("Resume Service", file_path)
    app.include_router(service.router) #, prefix="/api")
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    uvicorn.run(app, host="localhost", port=8000)
