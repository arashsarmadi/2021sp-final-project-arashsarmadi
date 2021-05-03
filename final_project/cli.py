import argparse
from pprint import pprint
import os

from canvasapi import Canvas


from csci_utils.canvas.canvas_helpers import CanvasHelper, pset_submission

from .pset_answers import get_answers

parser = argparse.ArgumentParser(description="Command description.")
parser.add_argument(
    "names", metavar="NAME", nargs=argparse.ZERO_OR_MORE, help="A name of something."
)


def main(args=None):
    args = parser.parse_args(args=args)

    # Start submission
    #
    # p_num = final-project
    # course_name = "Advanced Python for Data Science"
    # canvas_obj = Canvas(os.getenv("CANVAS_URL"), os.getenv("CANVAS_TOKEN"))
    #
    # can_help = CanvasHelper(course_name, canvas_obj)
    #
    # course_id = can_help.get_course_id(canvas_obj.get_courses())
    # course = canvas_obj.get_course(course_id)
    #
    # assignment_id = can_help.get_assignment_id(p_num, course.get_assignments())
    # quiz_id = can_help.get_quiz_id(p_num, course.get_quizzes())
    #
    # with pset_submission(
    #         canvas_obj, course_id, assignment_id, quiz_id
    # ) as q_sub:
    #     questions = q_sub.get_submission_questions()
    #
    #     # Get some basic info to help develop
    #     for q in questions:
    #         q_text = q.question_text.split("\n", 1)[0]
    #         q_name = q.question_name
    #         print(f"{q_name} - {q_text}")
    #
    #         # MC and some q's have 'answers' not 'answer'
    #         pprint(
    #             {
    #                 k: getattr(q, k, None)
    #                 for k in ["question_type", "id", "answer", "answers"]
    #             }
    #         )
    #
    #         print()
    #
    #     # Submit your answers
    #     answers = get_answers(questions)
    #     pprint(answers)
    #     q_sub.answer_submission_questions(quiz_questions=answers)
