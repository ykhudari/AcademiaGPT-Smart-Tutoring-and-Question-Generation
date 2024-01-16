import datetime
from pydantic import BaseModel, Field
from abc import ABC
from typing import List, Optional

DifficultyType = int


def execute_and_trace(code: str) -> bool:
    try:
        # Compile the code with a custom filename
        compiled_code = compile(code, 'code_to_evaluate', 'exec')
        exec(compiled_code)
        return True
    except Exception:
        # Display the traceback which will correctly reference the original lines
        formatted_traceback = traceback.format_exc().replace('<string>', 'code_to_evaluate')
        sys.stderr.write(formatted_traceback)
        return False


class QuestionBase(BaseModel, ABC):
    """
    Base class for all question types. 
    """
    question_text: str = Field(description=(
        "The main text of the question. Markdown formatted"
    ))
    difficulty: DifficultyType = Field(description=(
        "An integer from 1 to 3 representing how difficult "
        "the question should be. 1 is easiest. 3 is hardest"
    ))
    topics: List[str] = Field(description=(
        "A list of one or more topics that the question is testing"
    ))

    def _repr_markdown_(self):
        return repr(self)


class SingleSelection(QuestionBase):
    """
    Question where user is presented a prompt in `question_text` and 
    a list of `choices`. They are supposed to provide the single best
    answer (`solution`) as an integer, which is the index into `choices.

    All questions must have a minimum of 3 options

    Examples
    --------
    {
      "question_text": "What does `.loc` do?\n\nBelow is an example of how it might be used\n\n```python\ndf.loc[1995, \"NorthEast\"]\n```",
      "difficulty": 2,
      "topics": ["pandas", "loc", "indexing"],
      "choices": [
        "The `.loc` method allows a user to select rows/columns by name",
        "The `.loc` method allows a  user to select rows/columns by their position",
        "The `.loc` method is for aggregating data"
      ],
      "solution": 0
    }
    """
    choices: List[str] = Field(description=(
        "A list of markdown formatted strings representing "
        "the options for the student. Minimum of length 3"
    ))
    solution: int = Field(description=(
        "Index into choices representing correct answer. Zero based"
    ))

    def check(self, response):
        return self.solution == response

    def __repr__(self):
        out = f"{self.question_text}\n\n"
        for i, c in enumerate(self.choices):
            if i == self.solution:
                out += f"- [x] {c}\n"
            else:
                out += f"- [ ] {c}\n"
        return out


class ManySingleSelections(BaseModel):
    questions: List[SingleSelection]

class MultipleSelection(QuestionBase):
    """
    Question where user is presented a prompt in `question_text` and 
    a list of `choices`. They are supposed to provide all answers that
    apply (`solution`)

    All questions must have a minimum of 3 options

    Examples
    --------
    {
      "question_text": "What are some possible consequences of a learning rate that is too large?",
      "difficulty": 2,
      "topics": ["optimization", "gradient descent"],
      "choices": [
        "The algorithm never converges",
        "The algorithm becomes unstable",
        "Learning is stable, but very slow"
      ],
      "solution": [0, 1]
    }
    """
    choices: List[str] = Field(description=(
        "A list of markdown formatted strings representing "
        "the options for the student. Minimum length of 3."
    ))
    solution: List[int] = Field(description=(
        "List of indices into choices representing correct answers. Zero based"
    ))

    def check(self, response):
        return set(self.solution) == set(response) and len(response) == len(
            self.solution
        )

    def __repr__(self):
        out = f"{self.question_text}\n\n"
        for i, c in enumerate(self.choices):
            if i in self.solution:
                out += f"- [x] {c}\n"
            else:
                out += f"- [ ] {c}\n"
        return out

class Code(QuestionBase):
    """
    Question where user is presented a prompt in `question_text` and 
    given `starting_code`. They are then supposed to modify the `starting_code`
    to complete the question. After doing so the code will be verified by running
    the following template as if it were python code:

    ```python
    {setup_code}

    {student_response}

    {test_code}
    ```

    The test code should have `assert` statements that verify the correctness of
    the `student_response`

    Examples
    --------
    {
      "question_text": "How would you create a `DatetimeIndex` starting on January 1, 2022 and ending on June 1, 2022 with the values taking every hour in between?\n\nSave this to a variable called `dates`",
      "difficulty": 2,
      "topics": ["pandas", "dates"],
      "starting_code": "dates = ...",
      "solution": "dates = pd.date_range(\"2022-01-01\", \"2022-06-01\", freq=\"h\")",
      "setup_code": "import pandas as pd",
      "test_code": "assert dates.sort_values()[0].strftime(\"%Y-%m-%d\") == \"2022-01-01\"\nassert dates.sort_values()[-1].strftime(\"%Y-%m-%d\") == \"2022-06-01\"\nassert dates.shape[0] == 3625"
    }
    """
    starting_code: str = Field(description=(
        "Starting code that will be the initial contents of the "
        "student's text editor. Used to provide scaffold/skeleton code"
    ))
    solution: str = Field(description="The correct code")
    setup_code: str = Field(description=(
        "Any code that needs to execute prior to the student code to "
        "ensure any libraries are imported and any variables are set up"
    ))
    test_code: str = Field(description=(
        "Code containing `assert` statements that verifies the correctness"
        "of the student's response"
    ))

    def check(self, response):
        to_evaluate = f"{self.setup_code}\n\n{response}\n\n{self.test_code}\n\nTrue"
        return execute_and_trace(to_eval)

    def __repr__(self):
        out = f"{self.question_text}\n\n```python\n{self.starting_code}\n```"
        out += f"\n\n**Solution**\n\n```python\n{self.solution}\n```"
        out += f"\n\n**Test Suite**\n\n```python\n{self.setup_code}\n\n{self.solution}\n\n{self.test_code}\n```"
        return out
        

class FillInBlank(QuestionBase):
    """
    Question type where the student is given a main question and then
    a code block with "blanks" (represented by `___X` in the source).
    The student must provide one string per blank. Correctness is evaluated
    based on a Python test suite based on the following template:

    
    ```python
    {setup_code}

    {code_block_with_blanks_filled_in}

    {test_code}
    ```

    There must be at least one `___X` (one blank) in `starting_code`


    Examples
    --------
    {
      "question_text": "Suppose you have already executed the following code:\n\n```python\nimport numpy as np\n\nA = np.array([[1, 2], [3, 4]])\nb = np.array([10, 42])\n```\n\nFill in the blanks below to solve the matrix equation $Ax = b$ for $x$\n",
      "difficulty": 2,
      "topics": ["linear algebra", "regression", "numpy"],
      "starting_code": "from scipy.linalg import ___X\n\nx = ___X(A, ___X)",
      "solution": ["solve", "solve", "b"],
      "setup_code": "import numpy as np\n\nA = np.array([[1, 2], [3, 4]])\nb = np.array([10, 42])\n",
      "test_code": "assert np.allclose(x, [22, -6])"
    }
    """
    starting_code: str = Field(description=(
        " The starting code for the student. Must contain at least one "
        "`___X` (three underscores and capital `X`), which represents "
        "a blank that will be filled in by the student."
    ))
    solution: List[str] = Field(description=(
        "A list of strings representing the correct code to place in "
        "the blanks. Length must match number of `___X` that appear in "
        "`starting_code`."
    ))
    setup_code: str = Field(description=(
        "Any code that needs to execute prior to the student code to "
        "ensure any libraries are imported and any variables are set up"
    ))
    test_code: str = Field(description=(
        "Code containing `assert` statements that verifies the correctness "
        "of the student's response"
    ))

    def merge_answer(self, response: List[str]):
        parts = self.starting_code.split("___X")
        n_blanks = len(parts) - 1
        assert len(response) == n_blanks
        pieces = []
        for x, y in zip(parts, response + [""]):
            pieces.extend([x, y])
        return "".join(pieces)

    def check(self, response):
        code = self.merge_answer(response)
        to_eval = f"{self.setup_code}\n\n{code}\n\n{self.test_code}\n\nTrue"
        return execute_and_trace(to_eval)

    def __repr__(self):
        merged = self.merge_answer(self.solution)
        out = f"{self.question_text}\n\n```python\n{self.starting_code}\n```"
        out += f"\n\n**Solution**\n\n[{', '.join(self.solution)}]\n```"
        out += f"\n\n**Rendered Solution**\n\n```python\n{merged}\n```"
        out += f"\n\n**Test Suite**\n\n```python\n{self.setup_code}\n\n{merged}\n\n{self.test_code}\n```"
        return out