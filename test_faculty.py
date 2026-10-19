"""Regression tests for the faculty / HOD skill (no database or API key needed).   python test_faculty.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
import skills  # noqa: E402


def check(question, must_have=(), must_not=()):
    reply = skills.faculty(question)
    for s in must_have:
        assert s in reply, f"{question!r}: expected {s!r} in reply:\n{reply}"
    for s in must_not:
        assert s not in reply, f"{question!r}: did not expect {s!r} in reply:\n{reply}"


def main():
    # the bug: filler word 'so' + department name used to be treated as a person ('So Ctech')
    check("so who is the HOD of ctech", ["Dr. Niranjana G", "hod.ctech.ktr.et@srmist.edu.in"], ["So Ctech", "dr-so"])
    check("hod of mechanical", ["Dr. K. Suresh Kumar"])
    check("who is the head of department of civil engineering", ["Dr. P. R. Kannan Rajkumar"])
    check("hod of data science", ["Dr. Kavitha V"])
    # unknown department: honest answer, no invented person or profile link
    check("who is the HOD of ece", ["department list", "staff-finder"], ["Dr."])
    # person names: never fabricate a direct profile URL
    check("who is dr rajesh kumar", ["staff-finder", "Rajesh Kumar"], ["faculty/dr-", "faculty/rajesh"])
    check("faculty", ["staff-finder"])
    print("faculty skill: all checks passed")


if __name__ == "__main__":
    main()
