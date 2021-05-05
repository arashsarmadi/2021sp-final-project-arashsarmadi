import json
import os
from typing import Dict

from canvasapi import Canvas
from git import Repo

from csci_utils.canvas.canvas_helpers import CanvasHelper


def get_submission_comments(repo: Repo) -> Dict:
    """Get some info about this submission"""
    return dict(
        hexsha=repo.head.commit.hexsha[:8],
        submitted_from=repo.remotes.origin.url,
        dt=repo.head.commit.committed_datetime.isoformat(),
        branch=os.environ.get("TRAVIS_BRANCH", None),  # repo.active_branch.name,
        is_dirty=repo.is_dirty(),
        travis_url=os.environ.get("TRAVIS_BUILD_WEB_URL", None),
        final_project_link="https://arashsarmadi.github.io/2021sp-final-project-arashsarmadi/"
    )


if __name__ == "__main__":

    # Start submission - This assignment doesn't have a quiz so here we direclty submit the assignment

    repo = Repo(".")
    course_name = "Advanced Python for Data Science"
    canvas_obj = Canvas(os.getenv("CANVAS_URL"), os.getenv("CANVAS_TOKEN"))

    canvas_helper = CanvasHelper(course_name, canvas_obj)

    course_id = canvas_helper.get_course_id(canvas_obj.get_courses())
    course = canvas_obj.get_course(course_id)

    assignment_id = canvas_helper.get_assignment_id(
        "Final Project", course.get_assignments()
    )  # Entering the names directly since this assignment does not follow standard naming convention

    course = canvas_obj.get_course(course_id)
    assignment = course.get_assignment(assignment_id)

    # Begin submissions
    url = "https://github.com/csci-e-29/{}/commit/{}".format(
        os.path.basename(repo.working_dir), repo.head.commit.hexsha
    )  # you MUST push to the classroom org, even if CI/CD runs elsewhere
    # (you can push anytime before peer review begins)

    submission_comments = dict(text_comment=json.dumps(get_submission_comments(repo)))
    submission_dict = dict(submission_type="online_url", url=url)

    if repo.is_dirty():
        raise RuntimeError(
            "Must submit from a clean working directory - commit your code and rerun"
        )

    assignment.submit(submission_dict, comment=submission_comments)
